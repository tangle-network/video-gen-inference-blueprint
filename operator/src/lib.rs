pub mod config;
pub mod health;
pub mod qos;
pub mod video;

pub mod server;

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::{oneshot, Semaphore};

use crate::config::OperatorConfig;
use crate::video::{JobStatus, VideoBackend, VideoGenRequest};

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload ABI-encoded in the Tangle job call.
    struct VideoGenJobRequest {
        string prompt;
        /// Video duration in seconds
        uint32 durationSecs;
        /// Resolution as "WxH" (e.g. "1280x720")
        string resolution;
        /// Frames per second (0 = use operator default)
        uint32 fps;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload ABI-encoded in the Tangle job result.
    struct VideoGenJobResult {
        /// Job ID for polling/download
        string jobId;
        /// URL to download the generated video
        string outputUrl;
        /// Actual duration of generated video in seconds
        uint32 actualDurationSecs;
        /// Generation time in milliseconds
        uint64 generationTimeMs;
    }
}

// --- Job IDs ---

pub const VIDEO_GEN_JOB: u8 = 0;

// --- Shared state for the on-chain job handler ---

static VIDEO_BACKEND: std::sync::OnceLock<Arc<VideoBackend>> = std::sync::OnceLock::new();

/// Called by VideoGenServer to register the backend for on-chain job handlers.
#[allow(clippy::result_large_err)]
fn register_video_backend(backend: Arc<VideoBackend>) -> Result<(), RunnerError> {
    let _ = VIDEO_BACKEND.set(backend);
    Ok(())
}

/// Initialize the video backend for testing.
pub fn init_for_testing(config: Arc<OperatorConfig>) {
    let backend =
        Arc::new(VideoBackend::new(config).expect("failed to create video backend for testing"));
    let _ = VIDEO_BACKEND.set(backend);
}

/// Shared state for direct testing (raw HTTP, no VideoBackend).
static DIRECT_ENDPOINT: std::sync::OnceLock<DirectEndpoint> = std::sync::OnceLock::new();

struct DirectEndpoint {
    url: String,
    client: reqwest::Client,
}

/// Initialize a raw HTTP endpoint for direct testing (wiremock).
pub fn init_direct_for_testing(base_url: &str) {
    let _ = DIRECT_ENDPOINT.set(DirectEndpoint {
        url: format!("{base_url}/v1/video/generate"),
        client: reqwest::Client::new(),
    });
}

/// Direct video generation submit -- bypasses VideoBackend, does a raw HTTP POST.
/// Returns the job_id from the backend response.
pub async fn submit_video_direct(
    request: &VideoGenJobRequest,
) -> Result<String, RunnerError> {
    let endpoint = DIRECT_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other("direct endpoint not registered".into())
    })?;

    let body = serde_json::json!({
        "prompt": request.prompt,
        "duration_secs": request.durationSecs,
        "resolution": request.resolution,
        "fps": request.fps,
    });

    let resp = endpoint
        .client
        .post(&endpoint.url)
        .json(&body)
        .send()
        .await
        .map_err(|e| RunnerError::Other(format!("video gen request failed: {e}").into()))?;

    let result: serde_json::Value = resp.json().await
        .map_err(|e| RunnerError::Other(format!("video gen response parse failed: {e}").into()))?;

    let job_id = result["job_id"]
        .as_str()
        .ok_or_else(|| RunnerError::Other("missing job_id in response".into()))?
        .to_string();

    Ok(job_id)
}

// --- Router ---

pub fn router() -> Router {
    Router::new().route(
        VIDEO_GEN_JOB,
        run_video_gen
            .layer(TangleLayer)
            .layer(blueprint_sdk::tee::TeeLayer::new()),
    )
}

// --- Job handler ---

/// Handle a video generation job submitted on-chain.
///
/// Video generation is async and takes 30s-5min. This submits the job to the
/// backend, polls for completion, then returns the result. The Tangle job model
/// (submit -> process -> result) maps naturally to this pattern.
#[debug_job]
pub async fn run_video_gen(
    TangleArg(request): TangleArg<VideoGenJobRequest>,
) -> Result<TangleResult<VideoGenJobResult>, RunnerError> {
    let backend = VIDEO_BACKEND.get().ok_or_else(|| {
        RunnerError::Other("video backend not registered -- VideoGenServer not started".into())
    })?;

    let fps = if request.fps == 0 {
        backend.config.video.default_fps
    } else {
        request.fps
    };

    let gen_req = VideoGenRequest {
        prompt: request.prompt,
        duration_secs: request.durationSecs,
        resolution: request.resolution,
        fps,
    };

    // Submit job
    let job_id = backend.submit_job(gen_req).await.map_err(|e| {
        tracing::error!(error = %e, "video gen job submission failed");
        RunnerError::Other(format!("video gen submission failed: {e}").into())
    })?;

    // Poll for completion (video gen takes 30s-5min)
    let timeout = Duration::from_secs(backend.config.video.job_timeout_secs);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            return Err(RunnerError::Other(
                format!("video gen job {job_id} timed out").into(),
            ));
        }

        tokio::time::sleep(Duration::from_secs(3)).await;

        if let Some(job) = backend.get_job(&job_id) {
            match job.status {
                JobStatus::Completed => {
                    return Ok(TangleResult(VideoGenJobResult {
                        jobId: job_id,
                        outputUrl: job.output_url.unwrap_or_default(),
                        actualDurationSecs: job.duration_secs,
                        generationTimeMs: job.generation_time_ms.unwrap_or(0),
                    }));
                }
                JobStatus::Failed => {
                    let err = job.error.unwrap_or_else(|| "unknown error".to_string());
                    return Err(RunnerError::Other(
                        format!("video gen failed: {err}").into(),
                    ));
                }
                _ => continue,
            }
        }
    }
}

// --- Background service: Video backend + HTTP server ---

/// Runs the video generation backend and HTTP API as a [`BackgroundService`].
/// Starts before the BlueprintRunner begins polling for on-chain jobs.
#[derive(Clone)]
pub struct VideoGenServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for VideoGenServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Create video backend
            let backend = match VideoBackend::new(config.clone()) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create video backend");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            tracing::info!(
                mode = ?config.video.mode,
                endpoint = %config.video.endpoint,
                "video backend created, waiting for readiness"
            );

            if let Err(e) = backend.wait_ready().await {
                tracing::error!(error = %e, "video backend failed to become ready");
                let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                return;
            }
            tracing::info!("video backend is ready");

            // Register for on-chain job handlers
            if let Err(e) = register_video_backend(backend.clone()) {
                tracing::error!(error = %e, "failed to register video backend");
                let _ = tx.send(Err(e));
                return;
            }

            // 2. Build semaphore (low concurrency for GPU-bound video gen)
            let max_concurrent = config.server.max_concurrent_jobs;
            let semaphore = Arc::new(if max_concurrent == 0 {
                Semaphore::new(Semaphore::MAX_PERMITS)
            } else {
                Semaphore::new(max_concurrent)
            });

            // 3. Shutdown channel
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 4. Start the HTTP server
            let state = server::AppState {
                config: config.clone(),
                backend: backend.clone(),
                semaphore,
            };

            match server::start(state, shutdown_rx).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started -- background service ready");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // 5. Watchdog: monitor backend health
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("received shutdown signal");
                        let _ = shutdown_tx.send(true);
                        return;
                    }
                }

                if !backend.is_healthy().await {
                    tracing::warn!("video backend health check failed");
                }
            }
        });

        Ok(rx)
    }
}
