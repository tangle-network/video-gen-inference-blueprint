use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use video_gen_inference::config::OperatorConfig;
use video_gen_inference::detect_gpus;
use video_gen_inference::VideoGenServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

/// Build ABI-encoded registration payload for VideoGenBSM.onRegister.
/// Format: abi.encode(string model, uint32 gpuCount, uint32 totalVramMib, string gpuModel, uint32 maxDurationSecs, string[] supportedResolutions, string endpoint)
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

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    let env = BlueprintEnvironment::load()?;

    // Registration mode: emit registration inputs and exit
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            model = %config.video.model,
            max_duration_secs = config.video.max_duration_secs,
            "Registration payload saved"
        );
        return Ok(());
    }

    // GPU detection (non-fatal)
    match detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }

            // Warn if insufficient VRAM for video generation
            let total_vram: u32 = gpus.iter().map(|g| g.memory_total_mib).sum();
            if total_vram < config.gpu.min_vram_mib {
                tracing::warn!(
                    total_vram_mib = total_vram,
                    required_mib = config.gpu.min_vram_mib,
                    "insufficient VRAM for video generation"
                );
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed");
        }
    }

    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service ID configured".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

    // QoS heartbeat
    let qos_enabled = config
        .qos
        .as_ref()
        .map(|q| q.heartbeat_interval_secs > 0)
        .unwrap_or(false);
    if qos_enabled {
        match video_gen_inference::qos::start_heartbeat(config.clone()).await {
            Ok(_handle) => {
                let interval = config.qos.as_ref().unwrap().heartbeat_interval_secs;
                tracing::info!(interval_secs = interval, "QoS heartbeat started");
            }
            Err(e) => {
                tracing::warn!(error = %e, "QoS heartbeat failed to start (disabled)");
            }
        }
    } else {
        tracing::info!("QoS heartbeat disabled");
    }

    let video_server = VideoGenServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(video_gen_inference::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(video_server)
        .run()
        .await?;

    Ok(())
}
