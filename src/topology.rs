use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SmtGroup {
    pub vcpus: Vec<u32>,
    pub numa_node: u32,
}

#[derive(Debug, Clone)]
pub struct Pair {
    pub runtime_vcpu: u32,
    pub sqpoll_vcpu: u32,
    pub numa_node: u32,
}

fn read_numa_map() -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    let dir = match fs::read_dir("/sys/devices/system/node") {
        Ok(d) => d,
        Err(_) => return map,
    };
    for entry in dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("node") {
            continue;
        }
        let node_id: u32 = match name[4..].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let cpulist_path = entry.path().join("cpulist");
        let raw = match fs::read_to_string(&cpulist_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Ok(cpus) = parse_cpu_list(&raw) {
            for c in cpus {
                map.insert(c, node_id);
            }
        }
    }
    map
}

pub fn read_smt_groups() -> io::Result<Vec<SmtGroup>> {
    let numa = read_numa_map();
    let present = parse_cpu_list(&fs::read_to_string("/sys/devices/system/cpu/present")?)?;
    let mut groups: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for cpu in present {
        let path = format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
            cpu
        );
        if !Path::new(&path).exists() {
            continue;
        }
        let raw = fs::read_to_string(&path)?;
        let siblings = parse_cpu_list(&raw)?;
        if siblings.is_empty() {
            continue;
        }
        let key = *siblings.iter().min().unwrap();
        groups.entry(key).or_insert(siblings);
    }
    Ok(groups
        .into_iter()
        .map(|(key, mut vcpus)| {
            vcpus.sort_unstable();
            let numa_node = numa.get(&key).copied().unwrap_or(0);
            SmtGroup { vcpus, numa_node }
        })
        .collect())
}

fn parse_cpu_list(s: &str) -> io::Result<Vec<u32>> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: u32 = lo
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;
            let hi: u32 = hi
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;
            for v in lo..=hi {
                out.push(v);
            }
        } else {
            let v: u32 = part
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;
            out.push(v);
        }
    }
    Ok(out)
}

/// Plan `pairs_wanted` (runtime, sqpoll) pairs grouped within each NUMA node.
///
/// Within a NUMA node, physical cores are split into adjacent chunks of two:
/// one runtime phys core + one sqpoll phys core. Each phys core contributes
/// both SMT siblings, so a chunk produces 2 pairs:
///   - pair_a: runtime on rt_core.vcpus[0], sqpoll on sq_core.vcpus[0]
///   - pair_b: runtime on rt_core.vcpus[1], sqpoll on sq_core.vcpus[1]
///
/// Runtime ↔ sqpoll never share a physical core (no SMT contention between
/// userspace and kthread). Same-role siblings share a physical core (cheap).
pub fn plan_pairs(
    allowed_vcpus: &[u32],
    smt_groups: &[SmtGroup],
    pairs_wanted: usize,
) -> Result<Vec<Pair>, String> {
    let allowed: std::collections::HashSet<u32> = allowed_vcpus.iter().copied().collect();
    let usable: Vec<&SmtGroup> = smt_groups
        .iter()
        .filter(|g| g.vcpus.iter().all(|v| allowed.contains(v)))
        .collect();
    let mut by_node: BTreeMap<u32, Vec<&SmtGroup>> = BTreeMap::new();
    for g in &usable {
        by_node.entry(g.numa_node).or_default().push(*g);
    }
    let mut pairs: Vec<Pair> = Vec::with_capacity(pairs_wanted);
    'outer: for (node, groups) in by_node {
        let mut iter = groups.into_iter();
        loop {
            if pairs.len() == pairs_wanted {
                break 'outer;
            }
            let rt_group = match iter.next() {
                Some(g) => g,
                None => break,
            };
            let sq_group = match iter.next() {
                Some(g) => g,
                None => break,
            };
            let n_in_chunk = rt_group.vcpus.len().min(sq_group.vcpus.len());
            for i in 0..n_in_chunk {
                if pairs.len() == pairs_wanted {
                    break 'outer;
                }
                pairs.push(Pair {
                    runtime_vcpu: rt_group.vcpus[i],
                    sqpoll_vcpu: sq_group.vcpus[i],
                    numa_node: node,
                });
            }
        }
    }
    if pairs.len() < pairs_wanted {
        return Err(format!(
            "could not place {} runtime/SQPOLL pairs (got {}). Each NUMA node needs pairs of phys cores (1 runtime phys core + 1 sqpoll phys core = 2 pairs); cpuset may exclude some.",
            pairs_wanted, pairs.len()
        ));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(vcpus: &[u32], numa_node: u32) -> SmtGroup {
        SmtGroup {
            vcpus: vcpus.to_vec(),
            numa_node,
        }
    }

    #[test]
    fn parse_list_basic() {
        assert_eq!(parse_cpu_list("0,8").unwrap(), vec![0, 8]);
        assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0-1,4,6-7").unwrap(), vec![0, 1, 4, 6, 7]);
    }

    #[test]
    fn plan_four_pairs_one_ccx() {
        // 4 phys cores in one NUMA node, SMT siblings 32 apart.
        // Expect 4 pairs: cores 0,1 = runtime phys cores; cores 2,3 = sqpoll phys cores.
        let groups = vec![
            g(&[0, 32], 0),
            g(&[1, 33], 0),
            g(&[2, 34], 0),
            g(&[3, 35], 0),
        ];
        let allowed: Vec<u32> = (0..4).chain(32..36).collect();
        let pairs = plan_pairs(&allowed, &groups, 4).unwrap();
        assert_eq!(pairs.len(), 4);
        // Chunk 0: rt_core (0,32) + sq_core (1,33) -> pair0=(0,1), pair1=(32,33)
        assert_eq!((pairs[0].runtime_vcpu, pairs[0].sqpoll_vcpu), (0, 1));
        assert_eq!((pairs[1].runtime_vcpu, pairs[1].sqpoll_vcpu), (32, 33));
        // Chunk 1: rt_core (2,34) + sq_core (3,35) -> pair2=(2,3), pair3=(34,35)
        assert_eq!((pairs[2].runtime_vcpu, pairs[2].sqpoll_vcpu), (2, 3));
        assert_eq!((pairs[3].runtime_vcpu, pairs[3].sqpoll_vcpu), (34, 35));
        for p in &pairs {
            assert_eq!(p.numa_node, 0);
        }
    }

    #[test]
    fn plan_spans_numa_nodes_in_order() {
        // 2 NUMA nodes, 2 phys cores each. Each node → 2 pairs.
        let groups = vec![
            g(&[0, 32], 0),
            g(&[1, 33], 0),
            g(&[4, 36], 1),
            g(&[5, 37], 1),
        ];
        let allowed: Vec<u32> = vec![0, 1, 4, 5, 32, 33, 36, 37];
        let pairs = plan_pairs(&allowed, &groups, 4).unwrap();
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0].numa_node, 0);
        assert_eq!(pairs[1].numa_node, 0);
        assert_eq!(pairs[2].numa_node, 1);
        assert_eq!(pairs[3].numa_node, 1);
        assert_eq!((pairs[0].runtime_vcpu, pairs[0].sqpoll_vcpu), (0, 1));
        assert_eq!((pairs[2].runtime_vcpu, pairs[2].sqpoll_vcpu), (4, 5));
    }

    #[test]
    fn plan_rejects_odd_phys_cores_in_node() {
        // Only 1 phys core in node 0: can't form a (rt, sq) chunk.
        let groups = vec![g(&[0, 32], 0)];
        let allowed: Vec<u32> = vec![0, 32];
        let err = plan_pairs(&allowed, &groups, 1).unwrap_err();
        assert!(err.contains("could not place"));
    }

    #[test]
    fn plan_rejects_when_cpuset_excludes() {
        let groups = vec![g(&[0, 32], 0), g(&[1, 33], 0)];
        // cpuset excludes the SMT siblings.
        let allowed: Vec<u32> = vec![0, 1];
        let err = plan_pairs(&allowed, &groups, 1).unwrap_err();
        assert!(err.contains("could not place"));
    }
}
