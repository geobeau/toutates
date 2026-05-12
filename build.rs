fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::Config::new()
        .out_dir("src/grpc")
        .bytes([".inference.ModelInferRequest.raw_input_contents"])
        .service_generator(Box::new(pajamax_build::PajamaxGen::Local))
        .compile_protos(
            &["proto/grpc_service.proto", "proto/model_config.proto"],
            &["proto"],
        )?;
    Ok(())
}
