use blueprint_std::sync::Arc;
use blueprint_std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::config::OperatorConfig;
use crate::health;
use crate::video::{Img2VidRequest, InterpolateRequest, UpscaleRequest, VideoBackend, VideoGenRequest};

/// Shared application state for the HTTP server.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub backend: Arc<VideoBackend>,
    pub semaphore: Arc<Semaphore>,
}

/// Start the HTTP server with graceful shutdown support.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/video/generate", post(submit_video_job))
        .route("/v1/video/img2vid", post(submit_img2vid_job))
        .route("/v1/video/upscale", post(submit_upscale_job))
        .route("/v1/video/interpolate", post(submit_interpolate_job))
        .route("/v1/video/:job_id", get(get_video_job))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .layer(DefaultBodyLimit::max(
            state.config.server.max_request_body_bytes,
        ))
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
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
    /// Text prompt describing the desired video
    pub prompt: String,

    /// Desired video duration in seconds
    #[serde(default = "default_duration")]
    pub duration_secs: u32,

    /// Output resolution (e.g. "1280x720", "768x768")
    #[serde(default)]
    pub resolution: Option<String>,

    /// Frames per second (0 or omit = use operator default)
    #[serde(default)]
    pub fps: Option<u32>,

    /// SpendAuth for billing (required when billing_required is true)
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub service_id: u64,
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
}

#[derive(Debug, Deserialize)]
pub struct Img2VidHttpRequest {
    /// Base64-encoded image (PNG or JPEG)
    pub image: String,
    /// Motion/style prompt
    pub prompt: String,
    /// Desired output duration in seconds
    #[serde(default = "default_duration")]
    pub duration_secs: u32,
    /// Additional generation parameters
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct UpscaleHttpRequest {
    /// URL of the source video to upscale
    pub video_url: String,
    /// Target resolution as "WxH" (e.g. "3840x2160")
    pub target_resolution: String,
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct InterpolateHttpRequest {
    /// URL of the source video
    pub video_url: String,
    /// Target frames per second
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

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: String,
}

fn default_duration() -> u32 {
    4
}

fn error_response(status: StatusCode, message: String, error_type: &str, code: &str) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message,
            r#type: error_type.to_string(),
            code: code.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

// --- Handlers ---

async fn submit_video_job(
    State(state): State<AppState>,
    Json(req): Json<VideoGenerateRequest>,
) -> Response {
    // 1. Validate prompt
    if req.prompt.trim().is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "prompt cannot be empty".to_string(),
            "validation_error",
            "empty_prompt",
        );
    }

    // 2. Validate duration
    let max_duration = state.config.video.max_duration_secs;
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
    if req.duration_secs == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "duration_secs must be > 0".to_string(),
            "validation_error",
            "invalid_duration",
        );
    }

    // 3. Validate resolution
    let resolution = req
        .resolution
        .unwrap_or_else(|| state.config.video.default_resolution.clone());
    if !state
        .config
        .video
        .supported_resolutions
        .contains(&resolution)
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "resolution '{resolution}' not supported. Supported: {:?}",
                state.config.video.supported_resolutions
            ),
            "validation_error",
            "unsupported_resolution",
        );
    }

    let fps = req.fps.unwrap_or(state.config.video.default_fps);

    // 4. Enforce billing
    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return error_response(
            StatusCode::PAYMENT_REQUIRED,
            "SpendAuth required. Include spend_auth in request body.".to_string(),
            "billing_error",
            "payment_required",
        );
    }

    // 5. Validate SpendAuth amount covers the job cost
    if let Some(ref spend_auth) = req.spend_auth {
        let requested_amount: u64 = match spend_auth.amount.parse() {
            Ok(v) => v,
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid spend_auth amount".to_string(),
                    "billing_error",
                    "invalid_amount",
                );
            }
        };

        let estimated_cost = state.config.calculate_video_cost(req.duration_secs);
        if requested_amount < estimated_cost {
            return error_response(
                StatusCode::PAYMENT_REQUIRED,
                format!(
                    "spend authorization ({requested_amount}) is less than estimated cost ({estimated_cost}) for {duration}s of video",
                    duration = req.duration_secs,
                ),
                "billing_error",
                "insufficient_amount",
            );
        }

        let min_charge = state.config.billing.min_charge_amount;
        if min_charge > 0 && requested_amount < min_charge {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend authorization ({requested_amount}) below minimum charge ({min_charge})"
                ),
                "billing_error",
                "below_min_charge",
            );
        }
    }

    // 6. Acquire semaphore permit
    let _permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "server at capacity ({} concurrent jobs max)",
                    state.config.server.max_concurrent_jobs
                ),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    // 7. Check backend health before accepting job
    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "video generation backend is unavailable".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    // 8. Submit the job
    let gen_req = VideoGenRequest {
        prompt: req.prompt,
        duration_secs: req.duration_secs,
        resolution: resolution.clone(),
        fps,
    };

    match state.backend.submit_job(gen_req).await {
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
    Json(req): Json<Img2VidHttpRequest>,
) -> Response {
    if !state.config.video.supported_operations.contains(&"img2vid".to_string()) {
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

    if image_bytes.len() > state.config.video.max_input_size_bytes {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "image size ({} bytes) exceeds maximum ({})",
                image_bytes.len(),
                state.config.video.max_input_size_bytes
            ),
            "validation_error",
            "input_too_large",
        );
    }

    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return error_response(
            StatusCode::PAYMENT_REQUIRED,
            "SpendAuth required. Include spend_auth in request body.".to_string(),
            "billing_error",
            "payment_required",
        );
    }

    let _permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "server at capacity ({} concurrent jobs max)",
                    state.config.server.max_concurrent_jobs
                ),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "video backend is unavailable".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    let gen_req = Img2VidRequest {
        image_bytes,
        prompt: req.prompt,
        duration_secs: req.duration_secs,
        params: req.params.unwrap_or(serde_json::Value::Null),
    };

    match state.backend.img2vid(gen_req).await {
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
        Err(e) => {
            tracing::error!(error = %e, "failed to submit img2vid job");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to submit job: {e}"),
                "internal_error",
                "job_submission_failed",
            )
        }
    }
}

async fn submit_upscale_job(
    State(state): State<AppState>,
    Json(req): Json<UpscaleHttpRequest>,
) -> Response {
    if !state.config.video.supported_operations.contains(&"upscale".to_string()) {
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

    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return error_response(
            StatusCode::PAYMENT_REQUIRED,
            "SpendAuth required. Include spend_auth in request body.".to_string(),
            "billing_error",
            "payment_required",
        );
    }

    let _permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "server at capacity ({} concurrent jobs max)",
                    state.config.server.max_concurrent_jobs
                ),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "video backend is unavailable".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    let upscale_req = UpscaleRequest {
        video_url: req.video_url,
        target_resolution: req.target_resolution.clone(),
    };

    match state.backend.upscale(upscale_req).await {
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
        Err(e) => {
            tracing::error!(error = %e, "failed to submit upscale job");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to submit job: {e}"),
                "internal_error",
                "job_submission_failed",
            )
        }
    }
}

async fn submit_interpolate_job(
    State(state): State<AppState>,
    Json(req): Json<InterpolateHttpRequest>,
) -> Response {
    if !state.config.video.supported_operations.contains(&"interpolate".to_string()) {
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

    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return error_response(
            StatusCode::PAYMENT_REQUIRED,
            "SpendAuth required. Include spend_auth in request body.".to_string(),
            "billing_error",
            "payment_required",
        );
    }

    let _permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "server at capacity ({} concurrent jobs max)",
                    state.config.server.max_concurrent_jobs
                ),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    if !state.backend.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "video backend is unavailable".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    let interp_req = InterpolateRequest {
        video_url: req.video_url,
        target_fps: req.target_fps,
    };

    match state.backend.interpolate(interp_req).await {
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
        Err(e) => {
            tracing::error!(error = %e, "failed to submit interpolate job");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to submit job: {e}"),
                "internal_error",
                "job_submission_failed",
            )
        }
    }
}

async fn get_video_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Response {
    match state.backend.get_job(&job_id) {
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
    let gpu_info = health::detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "model": state.config.video.model,
        "mode": format!("{:?}", state.config.video.mode),
        "capabilities": {
            "max_duration_secs": state.config.video.max_duration_secs,
            "supported_resolutions": state.config.video.supported_resolutions,
            "supported_operations": state.config.video.supported_operations,
            "max_input_size_bytes": state.config.video.max_input_size_bytes,
            "default_fps": state.config.video.default_fps,
            "default_resolution": state.config.video.default_resolution,
        },
        "pricing": {
            "price_per_second": state.config.billing.price_per_second,
            "currency": "tsUSD",
        },
        "gpu": {
            "count": state.config.gpu.expected_gpu_count,
            "min_vram_mib": state.config.gpu.min_vram_mib,
            "model": state.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_jobs": state.config.server.max_concurrent_jobs,
        },
        "active_jobs": state.backend.active_job_count(),
        "billing_required": state.config.billing.billing_required,
    }))
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let backend_healthy = state.backend.is_healthy().await;

    if backend_healthy {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "model": state.config.video.model,
            "active_jobs": state.backend.active_job_count(),
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn gpu_health() -> Result<Json<Vec<health::GpuInfo>>, (StatusCode, String)> {
    match health::detect_gpus().await {
        Ok(gpus) => Ok(Json(gpus)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}
