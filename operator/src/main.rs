use alloy_sol_types::SolValue;
use video_gen_inference::config::OperatorConfig;
use video_gen_inference::VideoGenServer;

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);
    let max_duration = config.video.max_duration_secs;
    let resolutions = config.video.supported_resolutions.clone();

    (
        config.video.model.clone(),
        gpu_count,
        total_vram,
        gpu_model,
        max_duration,
        resolutions,
        endpoint,
    )
        .abi_encode()
}

tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    video_gen_inference::router(),
    |config| VideoGenServer { config },
    qos
);
