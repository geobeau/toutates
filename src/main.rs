#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cli;
mod grpc;
mod loader;
mod metrics;
mod model_repository;
mod model_runtime;
mod scheduler;
mod tensor;
mod topology;
mod tracing;
use ::tracing::info;
use arc_swap::ArcSwap;
use clap::Parser;
use pajamax::{serve, Server};
use std::sync::Arc;

use hashbrown::HashMap;
use tracing_subscriber::EnvFilter;

use ort::{environment::GlobalThreadPoolOptions, execution_providers::{ArbitrarilyConfigurableExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch, TensorRTExecutionProvider}};

use crate::{
    grpc::{inference::GrpcInferenceServiceServer, TritonService},
    model_repository::{LoadedModel, LocalModelRepository, ModelRepository},
    model_runtime::{LoadModelRequest, ModelRuntimeManager, SessionStarter},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_file(true)
        .with_line_number(true)
        .with_target(true)
        .init();

    let args = cli::Args::parse();
    let processing_cores = args.processing_cores;
    let executor_cores = args.executor_cores;
    let pin_cpus: Option<Vec<core_affinity::CoreId>> = if args.cpu_pinning {
        let cores = core_affinity::get_core_ids().expect("failed to read cpuset");
        info!("CPU pinning enabled, available cores: {:?}", cores.iter().map(|c| c.id).collect::<Vec<_>>());
        Some(cores)
    } else {
        None
    };

    // SQPOLL pair layout: one runtime + its dedicated SQPOLL kthread per pair.
    // Pairs are grouped within NUMA nodes; runtime and sqpoll never share a
    // physical core (no SMT contention between userspace and kthread).
    let sqpoll_pairs: Option<Vec<topology::Pair>> = if args.sqpoll_enabled {
        if !args.cpu_pinning {
            panic!("--sqpoll-enabled requires --cpu-pinning");
        }
        let cpus = pin_cpus.as_ref().unwrap();
        let allowed: Vec<u32> = cpus.iter().map(|c| c.id as u32).collect();
        let smt = topology::read_smt_groups().expect("failed to read /sys cpu topology");
        let pairs = topology::plan_pairs(&allowed, &smt, processing_cores)
            .unwrap_or_else(|e| panic!("SQPOLL pair planning failed: {e}"));
        for (i, p) in pairs.iter().enumerate() {
            info!(
                "SQPOLL pair {} (numa node {}): runtime vCPU {}, sqpoll kernel vCPU {}",
                i, p.numa_node, p.runtime_vcpu, p.sqpoll_vcpu
            );
        }
        info!(
            "SQPOLL enabled: {} runtimes, each with its own SQPOLL kernel thread",
            pairs.len()
        );
        Some(pairs)
    } else {
        None
    };

    // vCPUs reserved by SQPOLL pairs (runtime + kthread). Manager/executor/ORT
    // pin from whatever the cpuset has left.
    let reserved_vcpus: std::collections::HashSet<u32> = sqpoll_pairs
        .as_ref()
        .map(|ps| {
            ps.iter()
                .flat_map(|p| [p.runtime_vcpu, p.sqpoll_vcpu])
                .collect()
        })
        .unwrap_or_default();
    let unreserved_pin_cpus: Option<Vec<core_affinity::CoreId>> = pin_cpus.as_ref().map(|cpus| {
        cpus.iter()
            .copied()
            .filter(|c| !reserved_vcpus.contains(&(c.id as u32)))
            .collect()
    });
    // Manager + executor pin from the unreserved pool under SQPOLL; the existing
    // flat pin_cpus list otherwise.
    let mgr_exec_pin_pool: Option<&Vec<core_affinity::CoreId>> = if args.sqpoll_enabled {
        unreserved_pin_cpus.as_ref()
    } else {
        pin_cpus.as_ref()
    };
    let mut pin_index: usize = 0;

    // Validate CLI early so we fail fast before starting gRPC / workers
    let model_source = args.model_source();

    // Discover models before starting workers so errors surface immediately
    let discovered: Vec<LoadedModel> = match model_source {
        cli::ModelSource::Local(path) => {
            let repo = LocalModelRepository::new(path);
            repo.load_all()
                .expect("failed to load models from local directory")
        }
        cli::ModelSource::S3 {
            endpoint,
            bucket,
            prefix,
            region,
            cache_dir,
        } => {
            let repo = ModelRepository::new(&endpoint, &bucket, &prefix, &region, cache_dir);
            let filter: Option<std::collections::HashSet<String>> = args
                .load_models
                .as_ref()
                .map(|v| v.iter().cloned().collect());
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(repo.load_all(filter.as_ref()))
                .expect("failed to load models from S3")
        }
    };
    let discovered: Vec<LoadedModel> = if let Some(ref filter) = args.load_models {
        let allowed: std::collections::HashSet<&str> = filter.iter().map(|s| s.as_str()).collect();
        discovered
            .into_iter()
            .filter(|m| allowed.contains(m.name.as_str()))
            .collect()
    } else {
        discovered
    };

    let providers: Vec<ExecutionProviderDispatch> = args
        .execution_providers
        .iter()
        .map(|ep| match ep {
            cli::ExecutionProviderKind::Cpu => {
                info!("Registering CPU execution provider");
                ort::execution_providers::CPUExecutionProvider::default()
                    .build()
            }
            cli::ExecutionProviderKind::Cuda => {
                info!("Registering CUDA execution provider");
                CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build()
                    .error_on_failure()
            }
            cli::ExecutionProviderKind::TensorRT => {
                info!("Registering TensorRT execution provider");
                let mut ep = TensorRTExecutionProvider::default()
                    .with_device_id(0);
                if let Some(min_subgraph_size) = args.trt_min_subgraph_size {
                    ep = ep.with_arbitrary_config(
                        "trt_min_subgraph_size",
                        min_subgraph_size.to_string(),
                    );
                }
                if let Some(ref exclude) = args.trt_op_types_to_exclude {
                    ep = ep.with_arbitrary_config("trt_op_types_to_exclude", exclude.join(","));
                }
                ep.build()
                    .error_on_failure()
            }
        })
        .collect();
    // ORT's SetGlobalIntraOpThreadAffinity expects (thread_pool_size - 1) entries:
    // the calling thread is counted as the first member of the pool.
    // ORT uses 1-indexed processor IDs in the affinity string, so add 1 to each Linux CPU id.
    //
    // Under SQPOLL, processing threads no longer occupy a contiguous tail of
    // pin_cpus, so ORT picks from `unreserved_pin_cpus` starting after the
    // manager + executor slots. Without SQPOLL the legacy contiguous layout holds.
    let ort_intra_affinity: Option<String> = if args.ort_intra_threads <= 1 {
        None
    } else if args.sqpoll_enabled {
        unreserved_pin_cpus.as_ref().map(|cpus| {
            let ort_start = 1 + executor_cores;
            (0..args.ort_intra_threads - 1)
                .map(|i| (cpus[(ort_start + i) % cpus.len()].id + 1).to_string())
                .collect::<Vec<_>>()
                .join(";")
        })
    } else {
        pin_cpus.as_ref().map(|cpus| {
            let ort_start = 1 + executor_cores + processing_cores;
            (0..args.ort_intra_threads - 1)
                .map(|i| (cpus[(ort_start + i) % cpus.len()].id + 1).to_string())
                .collect::<Vec<_>>()
                .join(";")
        })
    };
    let mut thread_pool = GlobalThreadPoolOptions::default()
        .with_intra_threads(args.ort_intra_threads)
        .unwrap()
        .with_inter_threads(args.ort_inter_threads)
        .unwrap()
        .with_spin_control(false)
        .unwrap();
    if let Some(ref affinity) = ort_intra_affinity {
        thread_pool = thread_pool.with_intra_affinity(affinity).unwrap();
        info!("ORT intra-op thread affinity: {}", affinity);
    }
    ort::init()
        .with_execution_providers(providers)
        .with_global_thread_pool(thread_pool)
        .commit();

    // Metrics registry
    let metrics_registry = Arc::new(metrics::MetricsRegistry::new());

    // Shared model map for gRPC handlers
    let loaded_models: Arc<ArcSwap<HashMap<String, Arc<scheduler::ModelProxy>>>> =
        Arc::new(ArcSwap::from_pointee(HashMap::new()));

    // Create the load-model channel
    let (load_tx, load_rx) = tokio::sync::mpsc::channel::<LoadModelRequest>(16);

    let addr = &args.grpc_addr;
    info!("Starting Triton gRPC server on {}", addr);

    let config = pajamax::Config::new()
        .max_concurrent_connections(args.max_concurrent_connections)
        .max_concurrent_streams(args.max_concurrent_streams)
        .max_frame_size(args.max_frame_size)
        .buffer_pool_size(1024);

    let grpc_loaded_models = loaded_models.clone();
    let services: Vec<Box<dyn Fn() -> std::rc::Rc<dyn pajamax::PajamaxService> + Send + Sync>> =
        vec![Box::new(move || {
            std::rc::Rc::new(GrpcInferenceServiceServer::new(TritonService::new(
                grpc_loaded_models.clone(),
            )))
        })];
    let grpc_server = Server::new(services, config, addr.to_string());

    // Create per-executor-core session starter channels
    let mut starter_txs = Vec::with_capacity(executor_cores);
    let mut starter_rxs = Vec::with_capacity(executor_cores);
    for _ in 0..executor_cores {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        starter_txs.push(tx);
        starter_rxs.push(Some(rx));
    }

    // Spawn the ModelRuntimeManager on its own dedicated thread
    let custom_op_libraries = args.custom_op_libraries.unwrap_or_default();
    let manager_metrics = metrics_registry.clone();
    let manager = ModelRuntimeManager::new(
        load_rx,
        starter_txs,
        loaded_models,
        manager_metrics.clone(),
        custom_op_libraries,
    );
    let manager_pin_cpu = mgr_exec_pin_pool.map(|cpus| {
        let core = cpus[pin_index % cpus.len()];
        pin_index += 1;
        core
    });
    let manager_handle = std::thread::Builder::new()
        .name("model-manager".into())
        .spawn(move || {
            if let Some(core) = manager_pin_cpu {
                if !core_affinity::set_for_current(core) {
                    panic!("Failed to pin model-manager to CPU {}", core.id);
                }
                info!("Pinned model-manager to CPU {}", core.id);
            }
            let rt = compio::runtime::RuntimeBuilder::new()
                .build()
                .expect("failed to build manager compio runtime");
            rt.block_on(async move {
                metrics::init_local_metrics(manager_metrics);
                compio::runtime::spawn(async {
                    loop {
                        compio::time::sleep(std::time::Duration::from_secs(1)).await;
                        metrics::flush_local_metrics();
                    }
                })
                .detach();
                manager.run().await;
            });
        })
        .unwrap();

    let mut handles = vec![manager_handle];

    // Spawn dedicated executor core threads
    for core_id in 0..executor_cores {
        let starter_rx = starter_rxs[core_id].take().unwrap();
        let metrics_for_executor = metrics_registry.clone();
        let exec_pin_cpu = mgr_exec_pin_pool.map(|cpus| {
            let core = cpus[pin_index % cpus.len()];
            pin_index += 1;
            core
        });

        let handle = std::thread::Builder::new()
            .name(format!("exec-{core_id}"))
            .spawn(move || {
                if let Some(core) = exec_pin_cpu {
                    if !core_affinity::set_for_current(core) {
                        panic!("Failed to pin exec-{core_id} to CPU {}", core.id);
                    }
                    info!("Pinned exec-{core_id} to CPU {}", core.id);
                }
                let rt = compio::runtime::RuntimeBuilder::new()
                    .build()
                    .expect("failed to build executor compio runtime");
                rt.block_on(async move {
                    metrics::init_local_metrics(metrics_for_executor);
                    compio::runtime::spawn(async {
                        loop {
                            compio::time::sleep(std::time::Duration::from_secs(1)).await;
                            metrics::flush_local_metrics();
                        }
                    })
                    .detach();
                    SessionStarter::new(starter_rx).run().await;
                });
            })
            .unwrap();
        handles.push(handle);
    }

    // io_uring registered buffer pool (per-runtime, used by pajamax's read_multi).
    let buffer_pool_size = std::num::NonZero::new(args.buffer_pool_size)
        .expect("--buffer-pool-size must be > 0");
    let buffer_pool_buffer_len = args.buffer_pool_buffer_len;

    // Spawn gRPC processing core threads.
    if let Some(pairs) = sqpoll_pairs {
        // SQPOLL path: each runtime gets its own SQPOLL kernel thread pinned to a
        // dedicated vCPU. coop_taskrun/taskrun_flag are dropped (incompatible with
        // SQPOLL).
        let sqpoll_idle = std::time::Duration::from_millis(args.sqpoll_idle_ms as u64);
        for (idx, pair) in pairs.into_iter().enumerate() {
            let server = grpc_server.clone();
            let metrics_for_worker = metrics_registry.clone();
            let metrics_for_core = if idx == 0 {
                Some(metrics_registry.clone())
            } else {
                None
            };
            let runtime_vcpu = pair.runtime_vcpu;
            let sqpoll_vcpu = pair.sqpoll_vcpu;
            let handle = std::thread::Builder::new()
                .name(format!("proc-{idx}"))
                .spawn(move || {
                    let core = core_affinity::CoreId {
                        id: runtime_vcpu as usize,
                    };
                    if !core_affinity::set_for_current(core) {
                        panic!("Failed to pin proc-{idx} to CPU {runtime_vcpu}");
                    }
                    info!(
                        "Pinned proc-{idx} to CPU {runtime_vcpu} (own sqpoll on CPU {sqpoll_vcpu})"
                    );
                    let mut proactor = compio::driver::ProactorBuilder::new();
                    proactor
                        .capacity(8096)
                        .sqpoll_idle(sqpoll_idle)
                        .sqpoll_cpu(sqpoll_vcpu)
                        .buffer_pool_size(buffer_pool_size)
                        .buffer_pool_buffer_len(buffer_pool_buffer_len);
                    let rt = compio::runtime::RuntimeBuilder::new()
                        .with_proactor(proactor.to_owned())
                        .event_interval(1024)
                        .build()
                        .expect("failed to build SQPOLL compio runtime");
                    rt.block_on(async move {
                        metrics::init_local_metrics(metrics_for_worker);
                        compio::runtime::spawn(async {
                            loop {
                                compio::time::sleep(std::time::Duration::from_secs(1)).await;
                                metrics::flush_local_metrics();
                            }
                        })
                        .detach();
                        if let Some(mr) = metrics_for_core {
                            compio::runtime::spawn(metrics::serve_metrics("0.0.0.0:9090", mr))
                                .detach();
                        }
                        serve(server).await
                    })
                    .unwrap();
                })
                .unwrap();
            handles.push(handle);
        }
    } else {
        for core_id in 0..processing_cores {
            let server = grpc_server.clone();
            let metrics_for_worker = metrics_registry.clone();
            let metrics_for_core = if core_id == 0 {
                Some(metrics_registry.clone())
            } else {
                None
            };
            let proc_pin_cpu = pin_cpus.as_ref().map(|cpus| {
                let core = cpus[pin_index % cpus.len()];
                pin_index += 1;
                core
            });

            let handle = std::thread::Builder::new()
                .name(format!("proc-{core_id}"))
                .spawn(move || {
                    if let Some(core) = proc_pin_cpu {
                        if !core_affinity::set_for_current(core) {
                            panic!("Failed to pin proc-{core_id} to CPU {}", core.id);
                        }
                        info!("Pinned proc-{core_id} to CPU {}", core.id);
                    }
                    let mut proactor = compio::driver::ProactorBuilder::new();
                    proactor
                        .capacity(8096)
                        .coop_taskrun(true)
                        .taskrun_flag(true)
                        .buffer_pool_size(buffer_pool_size)
                        .buffer_pool_buffer_len(buffer_pool_buffer_len);

                    let rt = compio::runtime::RuntimeBuilder::new()
                        .with_proactor(proactor.to_owned())
                        .event_interval(1024)
                        .build()
                        .expect("failed to build compio runtime");

                    rt.block_on(async move {
                        metrics::init_local_metrics(metrics_for_worker);
                        compio::runtime::spawn(async {
                            loop {
                                compio::time::sleep(std::time::Duration::from_secs(1)).await;
                                metrics::flush_local_metrics();
                            }
                        })
                        .detach();

                        if let Some(mr) = metrics_for_core {
                            compio::runtime::spawn(metrics::serve_metrics("0.0.0.0:9090", mr)).detach();
                        }

                        serve(server).await
                    })
                    .unwrap();
                })
                .unwrap();
            handles.push(handle);
        }
    }

    // Dispatch discovered models to the runtime manager
    info!("Loading {} models", discovered.len());
    let mut load_replies = Vec::new();
    for model in discovered {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        load_tx.blocking_send(LoadModelRequest {
            model_name: model.name.clone(),
            version: model.version,
            model_path: model.model_path,
            config: model.config,
            reply: reply_tx,
        })?;
        load_replies.push((model.name, reply_rx));
    }
    for (name, reply_rx) in load_replies {
        match reply_rx.blocking_recv() {
            Ok(Ok(())) => info!("Model {} loaded successfully", name),
            Ok(Err(e)) => panic!("Failed to load model {}: {:?}", name, e),
            Err(_) => panic!("Model loader dropped without responding for {}", name),
        }
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }

    Ok(())
}
