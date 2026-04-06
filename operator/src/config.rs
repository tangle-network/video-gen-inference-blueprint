//! Video-gen-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

use blueprint_sdk::std::path::PathBuf;
use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// Video backend + per-second pricing configuration (video-gen-specific).
    pub video: VideoGenConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional — disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

/// Video generation backend mode.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoBackendMode {
    /// Local ComfyUI instance (Hunyuan Video / LTX-Video workflows).
    #[default]
    Local,
    /// Remote API (Modal, Replicate, or compatible endpoint).
    Api,
}

/// Video backend + pricing config. The only truly video-gen-specific config
/// section — everything else comes from `tangle-inference-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenConfig {
    /// Backend mode: "local" (ComfyUI) or "api" (Modal/Replicate).
    #[serde(default)]
    pub mode: VideoBackendMode,

    /// Video generation endpoint URL.
    pub endpoint: String,

    /// API key for remote backends (Modal/Replicate).
    #[serde(default)]
    pub api_key: Option<String>,

    /// Model identifier (e.g. "hunyuan-video", "ltx-video").
    pub model: String,

    /// Price per second of generated video in base token units.
    pub price_per_second: u64,

    /// Maximum video duration this operator supports (seconds).
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u32,

    /// Default frames per second for generated video.
    #[serde(default = "default_fps")]
    pub default_fps: u32,

    /// Supported output resolutions.
    #[serde(default = "default_supported_resolutions")]
    pub supported_resolutions: Vec<String>,

    /// Default resolution if not specified in request.
    #[serde(default = "default_resolution")]
    pub default_resolution: String,

    /// Timeout for a single video generation job (seconds).
    #[serde(default = "default_job_timeout_secs")]
    pub job_timeout_secs: u64,

    /// ComfyUI workflow template name (local mode only).
    #[serde(default)]
    pub comfyui_workflow: Option<String>,

    /// Directory to store generated video files.
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,

    /// Startup timeout for local backend (seconds).
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Operations this operator supports.
    #[serde(default = "default_supported_operations")]
    pub supported_operations: Vec<String>,

    /// Maximum input file size in bytes.
    #[serde(default = "default_max_input_size_bytes")]
    pub max_input_size_bytes: usize,
}

fn default_max_duration_secs() -> u32 {
    10
}

fn default_fps() -> u32 {
    24
}

fn default_supported_resolutions() -> Vec<String> {
    vec![
        "512x512".to_string(),
        "768x768".to_string(),
        "1280x720".to_string(),
    ]
}

fn default_resolution() -> String {
    "768x768".to_string()
}

fn default_job_timeout_secs() -> u64 {
    600
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("data/videos")
}

fn default_startup_timeout() -> u64 {
    300
}

fn default_supported_operations() -> Vec<String> {
    vec![
        "generate".to_string(),
        "img2vid".to_string(),
        "upscale".to_string(),
        "interpolate".to_string(),
    ]
}

fn default_max_input_size_bytes() -> usize {
    64 * 1024 * 1024
}

impl OperatorConfig {
    /// Load config from file and env vars.
    /// Env prefix: `VIDGEN_OP_` (e.g. `VIDGEN_OP_TANGLE__RPC_URL`).
    /// `VIDEO_ENDPOINT` overrides `video.endpoint`.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        builder = builder.add_source(
            config::Environment::with_prefix("VIDGEN_OP")
                .separator("__")
                .try_parsing(true),
        );

        if let Ok(endpoint) = std::env::var("VIDEO_ENDPOINT") {
            builder = builder.set_override("video.endpoint", endpoint)?;
        }

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "video": {
                "mode": "local",
                "endpoint": "http://127.0.0.1:8188",
                "model": "hunyuan-video",
                "price_per_second": 100000,
                "max_duration_secs": 10,
                "default_fps": 24
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "max_spend_per_request": 10000000,
                "min_credit_balance": 1000
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 49152
            }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.video.model, "hunyuan-video");
        assert_eq!(cfg.video.max_duration_secs, 10);
        assert_eq!(cfg.video.default_fps, 24);
        assert_eq!(cfg.video.price_per_second, 100000);
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        assert_eq!(cfg.video.job_timeout_secs, 600);
        assert_eq!(cfg.video.default_resolution, "768x768");
        assert_eq!(cfg.video.supported_resolutions.len(), 3);
        assert_eq!(cfg.video.supported_operations.len(), 4);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }
}
