use serde::{Deserialize, Serialize};
use blueprint_sdk::std::fmt;
use blueprint_sdk::std::path::PathBuf;

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration
    pub tangle: TangleConfig,

    /// Video generation backend configuration
    pub video: VideoConfig,

    /// HTTP server configuration
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration
    pub billing: BillingConfig,

    /// GPU configuration
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional)
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("video", &self.video)
            .field("server", &self.server)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain
    pub rpc_url: String,

    /// Chain ID
    pub chain_id: u64,

    /// Operator's private key (hex, without 0x prefix).
    /// In production, use a KMS or hardware signer instead.
    pub operator_key: String,

    /// Tangle core contract address
    pub tangle_core: String,

    /// ShieldedCredits contract address
    pub shielded_credits: String,

    /// Blueprint ID this operator is registered for
    pub blueprint_id: u64,

    /// Service ID (set after service activation)
    pub service_id: Option<u64>,
}

impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("tangle_core", &self.tangle_core)
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

/// Video generation backend mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoBackendMode {
    /// Local ComfyUI instance (Hunyuan Video / LTX-Video workflows)
    Local,
    /// Remote API (Modal, Replicate, or compatible endpoint)
    Api,
}

impl Default for VideoBackendMode {
    fn default() -> Self {
        Self::Local
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Backend mode: "local" (ComfyUI) or "api" (Modal/Replicate)
    #[serde(default)]
    pub mode: VideoBackendMode,

    /// Video generation endpoint URL.
    /// Local: ComfyUI API (e.g. http://127.0.0.1:8188)
    /// API: Modal/Replicate endpoint
    /// Can be overridden via VIDEO_ENDPOINT env var.
    pub endpoint: String,

    /// API key for remote backends (Modal/Replicate)
    pub api_key: Option<String>,

    /// Model identifier (e.g. "hunyuan-video", "ltx-video", "replicate/minimax/video-01")
    pub model: String,

    /// Maximum video duration this operator supports (seconds)
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u32,

    /// Default frames per second for generated video
    #[serde(default = "default_fps")]
    pub default_fps: u32,

    /// Supported output resolutions (e.g. ["512x512", "768x768", "1280x720"])
    #[serde(default = "default_supported_resolutions")]
    pub supported_resolutions: Vec<String>,

    /// Default resolution if not specified in request
    #[serde(default = "default_resolution")]
    pub default_resolution: String,

    /// Timeout for a single video generation job (seconds).
    /// Video gen is slow -- 30s to 5min+ depending on duration/resolution.
    #[serde(default = "default_job_timeout_secs")]
    pub job_timeout_secs: u64,

    /// ComfyUI workflow template name (local mode only)
    #[serde(default)]
    pub comfyui_workflow: Option<String>,

    /// Directory to store generated video files
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,

    /// Startup timeout for local backend (seconds)
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Operations this operator supports: "generate", "img2vid", "upscale", "interpolate"
    #[serde(default = "default_supported_operations")]
    pub supported_operations: Vec<String>,

    /// Maximum input file size in bytes (images, videos for processing)
    #[serde(default = "default_max_input_size_bytes")]
    pub max_input_size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    /// Maximum concurrent video generation jobs
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_jobs: usize,

    /// Maximum request body size in bytes (default 16 MiB)
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Per-account concurrent job limit. 0 = unlimited.
    #[serde(default)]
    pub max_per_account_jobs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Whether billing is required for HTTP requests
    #[serde(default = "default_billing_required")]
    pub required: bool,

    /// Price per second of generated video in tsUSD base units (6 decimals)
    pub price_per_second: u64,

    /// Maximum amount a single SpendAuth can authorize
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account
    pub min_credit_balance: u64,

    /// Whether billing (spend_auth) is required on every request
    #[serde(default = "default_billing_required")]
    pub billing_required: bool,

    /// Minimum charge amount per request (gas cost protection)
    #[serde(default)]
    pub min_charge_amount: u64,

    /// Maximum retries for claim_payment on-chain calls
    #[serde(default = "default_claim_max_retries")]
    pub claim_max_retries: u32,

    /// Clock skew tolerance in seconds for SpendAuth expiry checks
    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    /// Maximum gas price in gwei for billing txs. 0 = no cap.
    #[serde(default)]
    pub max_gas_price_gwei: u64,

    /// Path to persist used nonces across restarts
    #[serde(default = "default_nonce_store_path")]
    pub nonce_store_path: Option<PathBuf>,

    /// ERC-20 token address for x402 payment (e.g. tsUSD)
    #[serde(default)]
    pub payment_token_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Expected number of GPUs
    pub expected_gpu_count: u32,

    /// Minimum required VRAM per GPU in MiB.
    /// Hunyuan Video requires 48GB+, LTX-Video ~24GB.
    pub min_vram_mib: u32,

    /// GPU model name for on-chain registration
    #[serde(default)]
    pub gpu_model: Option<String>,

    /// GPU monitoring interval in seconds
    #[serde(default = "default_monitor_interval")]
    pub monitor_interval_secs: u64,
}

// --- Defaults ---

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
    600 // 10 minutes
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
    64 * 1024 * 1024 // 64 MiB
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_concurrent() -> usize {
    4 // video gen is GPU-heavy, low concurrency
}

fn default_billing_required() -> bool {
    true
}

fn default_monitor_interval() -> u64 {
    30
}

fn default_max_request_body_bytes() -> usize {
    16 * 1024 * 1024
}

fn default_claim_max_retries() -> u32 {
    3
}

fn default_clock_skew_tolerance() -> u64 {
    30
}

fn default_nonce_store_path() -> Option<PathBuf> {
    Some(PathBuf::from("data/nonces.json"))
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: VIDGEN_OP_ (e.g. VIDGEN_OP_TANGLE__RPC_URL)
        builder = builder.add_source(
            config::Environment::with_prefix("VIDGEN_OP")
                .separator("__")
                .try_parsing(true),
        );

        // VIDEO_ENDPOINT env var overrides video.endpoint
        if let Ok(endpoint) = std::env::var("VIDEO_ENDPOINT") {
            builder = builder.set_override("video.endpoint", endpoint)?;
        }

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }

    /// Calculate billing cost for a given video duration.
    pub fn calculate_video_cost(&self, duration_secs: u32) -> u64 {
        self.billing.price_per_second * duration_secs as u64
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
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "video": {
                "mode": "local",
                "endpoint": "http://127.0.0.1:8188",
                "model": "hunyuan-video",
                "max_duration_secs": 10,
                "default_fps": 24
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "price_per_second": 100000,
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
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.billing.price_per_second, 100000);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert_eq!(cfg.gpu.min_vram_mib, 49152);
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.server.max_concurrent_jobs, 4);
        assert_eq!(cfg.video.job_timeout_secs, 600);
        assert_eq!(cfg.video.default_resolution, "768x768");
        assert_eq!(cfg.video.supported_resolutions.len(), 3);
    }

    #[test]
    fn test_calculate_video_cost() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.calculate_video_cost(5), 500000);
        assert_eq!(cfg.calculate_video_cost(10), 1000000);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }
}
