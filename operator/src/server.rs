//! HTTP server for the video-gen operator.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! only contains the video-gen-specific HTTP handlers and request/response types.
//!
//! Because video generation is asynchronous (30s-5min per job), the billing
//! flow is different from the streaming/LLM path:
//!   1. Client POSTs to `/v1/video/generate` with an `x402` SpendAuth header.
//!   2. Operator validates the SpendAuth (signature, balance, nonce) and
//!      authorizes the spend on-chain **upfront** for the full requested
//!      duration (the client paid for the video they asked for).
//!   3. Operator returns a `job_id`; client polls `/v1/video/:job_id` for
//!      completion. Payment is claimed once the job finishes successfully.
//!   4. On failure, no payment is claimed (operator eats the gas of the
//!      upfront `authorizeSpend`).

use blueprint_sdk::std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    acquire_permit, billing_gate, error_response, gpu_health_handler, metrics_handler,
};
use tangle_inference_core::{detect_gpus, AppState, SpendAuthPayload};

use crate::video::{
    Img2VidRequest, InterpolateRequest, UpscaleRequest, VideoGenBackend, VideoGenRequest,
};

/// Start the HTTP server with graceful shutdown support.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<VideoGenBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not a VideoGenBackend"))?;
    let max_body = state.server_config.max_request_body_bytes;
    let stream_timeout = state.server_config.stream_timeout_secs;
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let _ = backend; // validated before spawning

    let app = HttpRouter::new()
        .route("/v1/video/generate", post(submit_video_job))
        .route("/v1/video/img2vid", post(submit_img2vid_job))
        .route("/v1/video/upscale", post(submit_upscale_job))
        .route("/v1/video/interpolate", post(submit_interpolate_job))
        .route("/v1/video/:job_id", get(get_video_job))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TimeoutLayer::new(Duration::from_secs(stream_timeout)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request / Response types ---

#[derive(Debug, Deserialize)]
pub struct VideoGenerateRequest {
    pub prompt: String,
    #[serde(default = "default_duration")]
    pub duration_secs: u32,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub fps: Option<u32>,
    /// Optional inline SpendAuth. Usually provided via `X-Payment-Signature`.
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct Img2VidHttpRequest {
    /// Base64-encoded image (PNG or JPEG).
    pub image: String,
    pub prompt: String,
    #[serde(default = "default_duration")]
    pub duration_secs: u32,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct UpscaleHttpRequest {
    pub video_url: String,
    pub target_resolution: String,
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct InterpolateHttpRequest {
    pub video_url: String,
    pub target_fps: u32,
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize)]
pub struct VideoGenerateResponse {
    pub job_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct VideoJobResponse {
    pub job_id: String,
    pub status: String,
    pub prompt: String,
    pub duration_secs: u32,
    pub resolution: String,
    pub fps: u32,
    pub output_url: Option<String>,
    pub error: Option<String>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub generation_time_ms: Option<u64>,
}

fn default_duration() -> u32 {
    4
}

// --- Helpers ---

fn backend_from(state: &AppState) -> &VideoGenBackend {
    state
        .backend::<VideoGenBackend>()
        .expect("AppState backend is VideoGenBackend (checked in server::start)")
}

// --- Handlers ---

async fn submit_video_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<VideoGenerateRequest>,
) -> Response {
    let backend = backend_from(&state);

    // Validate prompt
    if req.prompt.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "prompt cannot be empty".to_string(),
            "validation_error",
            "empty_prompt",
        );
    }

    // Validate duration
    let max_duration = backend.config.video.max_duration_secs;
    if req.duration_secs == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "duration_secs must be > 0".to_string(),
            "validation_error",
            "invalid_duration",
        );
    }
    if req.duration_secs > max_duration {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "duration_secs ({}) exceeds operator maximum ({max_duration})",
                req.duration_secs
            ),
            "validation_error",
            "duration_exceeded",
        );
    }

    // Validate resolution
    let resolution = req
        .resolution
        .clone()
        .unwrap_or_else(|| backend.config.video.default_resolution.clone());
    if !backend.config.video.supported_resolutions.contains(&resolution) {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "resolution '{resolution}' not supported. Supported: {:?}",
                backend.config.video.supported_resolutions
            ),
            "validation_error",
            "unsupported_resolution",
        );
    }

    let fps = req.fps.unwrap_or(backend.config.video.default_fps);

    // Concurrency gate
    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Billing gate — upfront authorize for the requested duration
    let estimated_cost = backend.calculate_cost(req.duration_secs);
    if let Err(resp) = billing_gate(&state, &headers, req.spend_auth, estimated_cost).await {
        return resp;
    }

    // Submit the job
    let gen_req = VideoGenRequest {
        prompt: req.prompt,
        duration_secs: req.duration_secs,
        resolution: resolution.clone(),
        fps,
    };

    match backend.submit_job(gen_req).await {
        Ok(job_id) => {
            tracing::info!(
                job_id = %job_id,
                duration_secs = req.duration_secs,
                resolution = %resolution,
                fps,
                "video generation job submitted"
            );
            Json(VideoGenerateResponse {
                job_id,
                status: "queued".to_string(),
                message: "Video generation job submitted. Poll GET /v1/video/:job_id for status."
                    .to_string(),
            })
            .into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "failed to submit video generation job");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to submit job: {e}"),
                "internal_error",
                "job_submission_failed",
            )
        }
    }
}

async fn submit_img2vid_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<Img2VidHttpRequest>,
) -> Response {
    let backend = backend_from(&state);

    if !backend
        .config
        .video
        .supported_operations
        .contains(&"img2vid".to_string())
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "img2vid operation not supported by this operator".to_string(),
            "validation_error",
            "unsupported_operation",
        );
    }

    if req.prompt.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "prompt cannot be empty".to_string(),
            "validation_error",
            "empty_prompt",
        );
    }

    let image_bytes = match base64::engine::general_purpose::STANDARD.decode(&req.image) {
        Ok(b) => b,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "image field must be valid base64".to_string(),
                "validation_error",
                "invalid_base64",
            );
        }
    };

    if image_bytes.len() > backend.config.video.max_input_size_bytes {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "image size ({} bytes) exceeds maximum ({})",
                image_bytes.len(),
                backend.config.video.max_input_size_bytes
            ),
            "validation_error",
            "input_too_large",
        );
    }

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let estimated_cost = backend.calculate_cost(req.duration_secs);
    if let Err(resp) = billing_gate(&state, &headers, req.spend_auth, estimated_cost).await {
        return resp;
    }

    let gen_req = Img2VidRequest {
        image_bytes,
        prompt: req.prompt,
        duration_secs: req.duration_secs,
        params: req.params.unwrap_or(serde_json::Value::Null),
    };

    match backend.img2vid(gen_req).await {
        Ok(job_id) => {
            tracing::info!(job_id = %job_id, "img2vid job submitted");
            Json(VideoGenerateResponse {
                job_id,
                status: "queued".to_string(),
                message: "Image-to-video job submitted. Poll GET /v1/video/:job_id for status."
                    .to_string(),
            })
            .into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to submit job: {e}"),
            "internal_error",
            "job_submission_failed",
        ),
    }
}

async fn submit_upscale_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UpscaleHttpRequest>,
) -> Response {
    let backend = backend_from(&state);

    if !backend
        .config
        .video
        .supported_operations
        .contains(&"upscale".to_string())
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "upscale operation not supported by this operator".to_string(),
            "validation_error",
            "unsupported_operation",
        );
    }

    if req.video_url.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "video_url cannot be empty".to_string(),
            "validation_error",
            "empty_video_url",
        );
    }

    if req.target_resolution.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "target_resolution cannot be empty".to_string(),
            "validation_error",
            "empty_target_resolution",
        );
    }

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let spend_auth = req.spend_auth;
    if let Err(resp) = billing_gate(&state, &headers, spend_auth, 0).await {
        return resp;
    }

    let upscale_req = UpscaleRequest {
        video_url: req.video_url,
        target_resolution: req.target_resolution.clone(),
    };

    match backend.upscale(upscale_req).await {
        Ok(job_id) => {
            tracing::info!(job_id = %job_id, target = %req.target_resolution, "upscale job submitted");
            Json(VideoGenerateResponse {
                job_id,
                status: "queued".to_string(),
                message: "Upscale job submitted. Poll GET /v1/video/:job_id for status."
                    .to_string(),
            })
            .into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to submit job: {e}"),
            "internal_error",
            "job_submission_failed",
        ),
    }
}

async fn submit_interpolate_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<InterpolateHttpRequest>,
) -> Response {
    let backend = backend_from(&state);

    if !backend
        .config
        .video
        .supported_operations
        .contains(&"interpolate".to_string())
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "interpolate operation not supported by this operator".to_string(),
            "validation_error",
            "unsupported_operation",
        );
    }

    if req.video_url.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "video_url cannot be empty".to_string(),
            "validation_error",
            "empty_video_url",
        );
    }

    if req.target_fps == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "target_fps must be > 0".to_string(),
            "validation_error",
            "invalid_target_fps",
        );
    }

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let spend_auth = req.spend_auth;
    if let Err(resp) = billing_gate(&state, &headers, spend_auth, 0).await {
        return resp;
    }

    let interp_req = InterpolateRequest {
        video_url: req.video_url,
        target_fps: req.target_fps,
    };

    match backend.interpolate(interp_req).await {
        Ok(job_id) => {
            tracing::info!(job_id = %job_id, target_fps = req.target_fps, "interpolate job submitted");
            Json(VideoGenerateResponse {
                job_id,
                status: "queued".to_string(),
                message: "Interpolation job submitted. Poll GET /v1/video/:job_id for status."
                    .to_string(),
            })
            .into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to submit job: {e}"),
            "internal_error",
            "job_submission_failed",
        ),
    }
}

async fn get_video_job(State(state): State<AppState>, Path(job_id): Path<String>) -> Response {
    let backend = backend_from(&state);
    match backend.get_job(&job_id) {
        Some(job) => Json(VideoJobResponse {
            job_id: job.job_id,
            status: serde_json::to_value(&job.status)
                .ok()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".to_string()),
            prompt: job.prompt,
            duration_secs: job.duration_secs,
            resolution: job.resolution,
            fps: job.fps,
            output_url: job.output_url,
            error: job.error,
            created_at: job.created_at,
            completed_at: job.completed_at,
            generation_time_ms: job.generation_time_ms,
        })
        .into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            format!("job '{job_id}' not found"),
            "not_found",
            "job_not_found",
        ),
    }
}

async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let gpu_info = detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": backend.config.video.model,
        "mode": format!("{:?}", backend.config.video.mode),
        "capabilities": {
            "max_duration_secs": backend.config.video.max_duration_secs,
            "supported_resolutions": backend.config.video.supported_resolutions,
            "supported_operations": backend.config.video.supported_operations,
            "max_input_size_bytes": backend.config.video.max_input_size_bytes,
            "default_fps": backend.config.video.default_fps,
            "default_resolution": backend.config.video.default_resolution,
        },
        "pricing": {
            "price_per_second": backend.config.video.price_per_second,
            "currency": "tsUSD",
        },
        "gpu": {
            "count": backend.config.gpu.expected_gpu_count,
            "min_vram_mib": backend.config.gpu.min_vram_mib,
            "model": backend.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_jobs": state.server_config.max_concurrent_requests,
        },
        "active_jobs": backend.active_job_count(),
        "billing_required": state.billing_config.billing_required,
        "payment_token": state.billing_config.payment_token_address,
    }))
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let backend = backend_from(&state);
    if backend.is_healthy().await {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": backend.config.video.model,
            "active_jobs": backend.active_job_count(),
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}
