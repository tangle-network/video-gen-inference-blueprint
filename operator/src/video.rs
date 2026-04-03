//! Video generation backend abstraction.
//!
//! Supports two modes:
//! - Local: ComfyUI with Hunyuan Video / LTX-Video workflows
//! - API: Remote endpoint (Modal, Replicate, or compatible)
//!
//! Both modes use an async job model: submit returns a job_id, poll for status.

use blueprint_std::sync::Arc;
use blueprint_std::time::Duration;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::config::{OperatorConfig, VideoBackendMode};

/// Status of a video generation job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job is queued, waiting for GPU
    Queued,
    /// Job is actively generating
    Processing,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
}

/// A video generation job tracked by the operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoJob {
    pub job_id: String,
    pub status: JobStatus,
    pub prompt: String,
    pub duration_secs: u32,
    pub resolution: String,
    pub fps: u32,
    /// Output video URL/path when completed
    pub output_url: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Timestamp when job was submitted (unix seconds)
    pub created_at: u64,
    /// Timestamp when job completed (unix seconds)
    pub completed_at: Option<u64>,
    /// Actual generation time in milliseconds
    pub generation_time_ms: Option<u64>,
}

/// Request to submit a video generation job to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenRequest {
    pub prompt: String,
    pub duration_secs: u32,
    pub resolution: String,
    pub fps: u32,
}

/// Request to convert a still image into a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Img2VidRequest {
    /// Raw image bytes (PNG/JPEG)
    #[serde(with = "base64_bytes")]
    pub image_bytes: Vec<u8>,
    /// Motion/style prompt
    pub prompt: String,
    /// Desired output duration in seconds
    pub duration_secs: u32,
    /// Additional generation parameters
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Request to upscale a video to higher resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleRequest {
    /// URL of the source video
    pub video_url: String,
    /// Target resolution as "WxH" (e.g. "3840x2160")
    pub target_resolution: String,
}

/// Request to interpolate frames (increase FPS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolateRequest {
    /// URL of the source video
    pub video_url: String,
    /// Target frames per second
    pub target_fps: u32,
}

/// Serde helper for base64-encoded byte fields.
mod base64_bytes {
    use serde::{Deserialize, Deserializer, Serializer};
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        STANDARD.decode(s).map_err(serde::de::Error::custom)
    }
}

/// The kind of operation a job is performing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Generate,
    Img2Vid,
    Upscale,
    Interpolate,
}

/// Video generation backend client.
pub struct VideoBackend {
    pub(crate) config: Arc<OperatorConfig>,
    client: reqwest::Client,
    /// In-memory job store. In production, back with persistent storage.
    jobs: DashMap<String, VideoJob>,
}

impl VideoBackend {
    pub fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.video.job_timeout_secs))
            .build()?;
        Ok(Self {
            config,
            client,
            jobs: DashMap::new(),
        })
    }

    /// Check if the video backend is healthy.
    pub async fn is_healthy(&self) -> bool {
        let health_url = match self.config.video.mode {
            VideoBackendMode::Local => format!("{}/system_stats", self.config.video.endpoint),
            VideoBackendMode::Api => format!("{}/health", self.config.video.endpoint),
        };
        matches!(
            self.client
                .get(&health_url)
                .timeout(Duration::from_secs(5))
                .send()
                .await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Wait for the backend to become ready (startup probe).
    pub async fn wait_ready(&self) -> anyhow::Result<()> {
        let timeout = Duration::from_secs(self.config.video.startup_timeout_secs);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!(
                    "video backend failed to become ready within {}s",
                    self.config.video.startup_timeout_secs
                );
            }
            if self.is_healthy().await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    /// Submit a video generation job. Returns immediately with a job_id.
    pub async fn submit_job(&self, req: VideoGenRequest) -> anyhow::Result<String> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let job = VideoJob {
            job_id: job_id.clone(),
            status: JobStatus::Queued,
            prompt: req.prompt.clone(),
            duration_secs: req.duration_secs,
            resolution: req.resolution.clone(),
            fps: req.fps,
            output_url: None,
            error: None,
            created_at: now,
            completed_at: None,
            generation_time_ms: None,
        };

        self.jobs.insert(job_id.clone(), job);

        // Dispatch to backend asynchronously
        let backend = self.clone_for_spawn();
        let job_id_clone = job_id.clone();
        let req_clone = req;
        tokio::spawn(async move {
            backend.execute_job(job_id_clone, req_clone).await;
        });

        Ok(job_id)
    }

    /// Get the current status of a job.
    pub fn get_job(&self, job_id: &str) -> Option<VideoJob> {
        self.jobs.get(job_id).map(|r| r.clone())
    }

    /// Get active job count.
    pub fn active_job_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|r| {
                matches!(r.value().status, JobStatus::Queued | JobStatus::Processing)
            })
            .count()
    }

    /// Submit an image-to-video job. Returns immediately with a job_id.
    pub async fn img2vid(&self, req: Img2VidRequest) -> anyhow::Result<String> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let job = VideoJob {
            job_id: job_id.clone(),
            status: JobStatus::Queued,
            prompt: req.prompt.clone(),
            duration_secs: req.duration_secs,
            resolution: String::new(),
            fps: 0,
            output_url: None,
            error: None,
            created_at: now,
            completed_at: None,
            generation_time_ms: None,
        };

        self.jobs.insert(job_id.clone(), job);

        let backend = self.clone_for_spawn();
        let jid = job_id.clone();
        tokio::spawn(async move {
            backend.execute_img2vid(jid, req).await;
        });

        Ok(job_id)
    }

    /// Submit a video upscale job. Returns immediately with a job_id.
    pub async fn upscale(&self, req: UpscaleRequest) -> anyhow::Result<String> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let job = VideoJob {
            job_id: job_id.clone(),
            status: JobStatus::Queued,
            prompt: String::new(),
            duration_secs: 0,
            resolution: req.target_resolution.clone(),
            fps: 0,
            output_url: None,
            error: None,
            created_at: now,
            completed_at: None,
            generation_time_ms: None,
        };

        self.jobs.insert(job_id.clone(), job);

        let backend = self.clone_for_spawn();
        let jid = job_id.clone();
        tokio::spawn(async move {
            backend.execute_upscale(jid, req).await;
        });

        Ok(job_id)
    }

    /// Submit a frame-interpolation job. Returns immediately with a job_id.
    pub async fn interpolate(&self, req: InterpolateRequest) -> anyhow::Result<String> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let job = VideoJob {
            job_id: job_id.clone(),
            status: JobStatus::Queued,
            prompt: String::new(),
            duration_secs: 0,
            resolution: String::new(),
            fps: req.target_fps,
            output_url: None,
            error: None,
            created_at: now,
            completed_at: None,
            generation_time_ms: None,
        };

        self.jobs.insert(job_id.clone(), job);

        let backend = self.clone_for_spawn();
        let jid = job_id.clone();
        tokio::spawn(async move {
            backend.execute_interpolate(jid, req).await;
        });

        Ok(job_id)
    }

    /// Execute a video generation job against the configured backend.
    async fn execute_job(&self, job_id: String, req: VideoGenRequest) {
        // Mark as processing
        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            job.status = JobStatus::Processing;
        }

        let start = std::time::Instant::now();

        let result = match self.config.video.mode {
            VideoBackendMode::Local => self.execute_local(&req).await,
            VideoBackendMode::Api => self.execute_api(&req).await,
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            match result {
                Ok(output_url) => {
                    job.status = JobStatus::Completed;
                    job.output_url = Some(output_url);
                    job.completed_at = Some(now);
                    job.generation_time_ms = Some(elapsed_ms);
                    tracing::info!(
                        job_id = %job_id,
                        elapsed_ms,
                        "video generation completed"
                    );
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.error = Some(e.to_string());
                    job.completed_at = Some(now);
                    job.generation_time_ms = Some(elapsed_ms);
                    tracing::error!(
                        job_id = %job_id,
                        error = %e,
                        "video generation failed"
                    );
                }
            }
        }
    }

    /// Execute via local ComfyUI backend.
    /// Submits a workflow prompt and polls for completion.
    async fn execute_local(&self, req: &VideoGenRequest) -> anyhow::Result<String> {
        let workflow = self.build_comfyui_workflow(req);

        // Submit workflow to ComfyUI
        let resp = self
            .client
            .post(format!("{}/prompt", self.config.video.endpoint))
            .json(&workflow)
            .send()
            .await?
            .error_for_status()?;

        let body: serde_json::Value = resp.json().await?;
        let prompt_id = body["prompt_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("ComfyUI response missing prompt_id"))?
            .to_string();

        // Poll for completion
        let poll_url = format!("{}/history/{}", self.config.video.endpoint, prompt_id);
        let timeout = Duration::from_secs(self.config.video.job_timeout_secs);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!("ComfyUI job timed out after {}s", timeout.as_secs());
            }

            tokio::time::sleep(Duration::from_secs(2)).await;

            let resp = self.client.get(&poll_url).send().await?;
            if !resp.status().is_success() {
                continue;
            }

            let history: serde_json::Value = resp.json().await?;
            if let Some(prompt_data) = history.get(&prompt_id) {
                if let Some(outputs) = prompt_data.get("outputs") {
                    // Extract the video file path from ComfyUI outputs
                    for (_node_id, node_output) in outputs
                        .as_object()
                        .unwrap_or(&serde_json::Map::new())
                    {
                        if let Some(videos) = node_output.get("videos") {
                            if let Some(first) = videos.as_array().and_then(|a| a.first()) {
                                let filename = first["filename"]
                                    .as_str()
                                    .unwrap_or("output.mp4");
                                let subfolder = first["subfolder"]
                                    .as_str()
                                    .unwrap_or("");
                                let output_url = format!(
                                    "{}/view?filename={}&subfolder={}&type=output",
                                    self.config.video.endpoint, filename, subfolder
                                );
                                return Ok(output_url);
                            }
                        }
                    }
                }

                // Check for errors
                if let Some(status) = prompt_data.get("status") {
                    if let Some(messages) = status.get("messages") {
                        if let Some(arr) = messages.as_array() {
                            for msg in arr {
                                if msg[0].as_str() == Some("execution_error") {
                                    let err = msg.get(1)
                                        .and_then(|v| v.get("exception_message"))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("unknown error");
                                    anyhow::bail!("ComfyUI execution error: {err}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Execute via remote API (Modal, Replicate, or compatible).
    /// Expects the endpoint to support: POST / -> {id}, GET /{id} -> {status, output}
    async fn execute_api(&self, req: &VideoGenRequest) -> anyhow::Result<String> {
        let mut submit_req = self
            .client
            .post(&self.config.video.endpoint)
            .json(&serde_json::json!({
                "prompt": req.prompt,
                "duration_secs": req.duration_secs,
                "resolution": req.resolution,
                "fps": req.fps,
                "model": self.config.video.model,
            }));

        if let Some(ref api_key) = self.config.video.api_key {
            submit_req = submit_req.bearer_auth(api_key);
        }

        let resp = submit_req.send().await?.error_for_status()?;
        let body: serde_json::Value = resp.json().await?;

        // Extract job ID from response (Replicate uses "id", Modal uses "call_id")
        let remote_job_id = body["id"]
            .as_str()
            .or_else(|| body["call_id"].as_str())
            .ok_or_else(|| anyhow::anyhow!("API response missing job id"))?
            .to_string();

        // Poll for completion
        let poll_url = format!("{}/{}", self.config.video.endpoint, remote_job_id);
        let timeout = Duration::from_secs(self.config.video.job_timeout_secs);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!("API job timed out after {}s", timeout.as_secs());
            }

            tokio::time::sleep(Duration::from_secs(3)).await;

            let mut poll_req = self.client.get(&poll_url);
            if let Some(ref api_key) = self.config.video.api_key {
                poll_req = poll_req.bearer_auth(api_key);
            }

            let resp = poll_req.send().await?.error_for_status()?;
            let body: serde_json::Value = resp.json().await?;

            let status = body["status"]
                .as_str()
                .unwrap_or("unknown");

            match status {
                "succeeded" | "completed" => {
                    let output_url = body["output"]
                        .as_str()
                        .or_else(|| body["output_url"].as_str())
                        .or_else(|| {
                            body["output"]
                                .as_array()
                                .and_then(|a| a.first())
                                .and_then(|v| v.as_str())
                        })
                        .ok_or_else(|| anyhow::anyhow!("API response missing output URL"))?
                        .to_string();
                    return Ok(output_url);
                }
                "failed" | "error" => {
                    let error_msg = body["error"]
                        .as_str()
                        .unwrap_or("unknown error");
                    anyhow::bail!("API job failed: {error_msg}");
                }
                _ => continue, // still processing
            }
        }
    }

    /// Build a ComfyUI workflow JSON for video generation.
    fn build_comfyui_workflow(&self, req: &VideoGenRequest) -> serde_json::Value {
        let (width, height) = parse_resolution(&req.resolution);
        let workflow_name = self
            .config
            .video
            .comfyui_workflow
            .as_deref()
            .unwrap_or("hunyuan_video_default");

        // Standard ComfyUI prompt API format.
        // This is a minimal workflow -- real deployments will use a full workflow JSON
        // loaded from a template file. The structure matches ComfyUI's /prompt endpoint.
        serde_json::json!({
            "prompt": {
                "1": {
                    "class_type": "HunyuanVideoSampler",
                    "inputs": {
                        "prompt": req.prompt,
                        "width": width,
                        "height": height,
                        "num_frames": req.fps * req.duration_secs,
                        "fps": req.fps,
                        "workflow": workflow_name,
                    }
                },
                "2": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "filename_prefix": "videogen",
                        "videos": ["1", 0],
                    }
                }
            }
        })
    }

    /// Execute an image-to-video job against the configured backend.
    async fn execute_img2vid(&self, job_id: String, req: Img2VidRequest) {
        self.execute_generic_job(job_id, "img2vid", || {
            let image_b64 = base64::engine::general_purpose::STANDARD.encode(&req.image_bytes);
            serde_json::json!({
                "operation": "img2vid",
                "image": image_b64,
                "prompt": req.prompt,
                "duration_secs": req.duration_secs,
                "params": req.params,
                "model": self.config.video.model,
            })
        }).await;
    }

    /// Execute a video upscale job against the configured backend.
    async fn execute_upscale(&self, job_id: String, req: UpscaleRequest) {
        self.execute_generic_job(job_id, "upscale", || {
            serde_json::json!({
                "operation": "upscale",
                "video_url": req.video_url,
                "target_resolution": req.target_resolution,
                "model": self.config.video.model,
            })
        }).await;
    }

    /// Execute a frame-interpolation job against the configured backend.
    async fn execute_interpolate(&self, job_id: String, req: InterpolateRequest) {
        self.execute_generic_job(job_id, "interpolate", || {
            serde_json::json!({
                "operation": "interpolate",
                "video_url": req.video_url,
                "target_fps": req.target_fps,
                "model": self.config.video.model,
            })
        }).await;
    }

    /// Generic job execution: mark processing, POST to backend, poll for result, update job store.
    async fn execute_generic_job<F>(&self, job_id: String, operation: &str, build_payload: F)
    where
        F: FnOnce() -> serde_json::Value,
    {
        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            job.status = JobStatus::Processing;
        }

        let start = std::time::Instant::now();
        let payload = build_payload();

        let result = async {
            let endpoint = &self.config.video.endpoint;
            let mut submit_req = self.client.post(endpoint).json(&payload);
            if let Some(ref api_key) = self.config.video.api_key {
                submit_req = submit_req.bearer_auth(api_key);
            }

            let resp = submit_req.send().await?.error_for_status()?;
            let body: serde_json::Value = resp.json().await?;

            let remote_id = body["id"]
                .as_str()
                .or_else(|| body["call_id"].as_str())
                .ok_or_else(|| anyhow::anyhow!("API response missing job id"))?
                .to_string();

            let poll_url = format!("{}/{}", endpoint, remote_id);
            let timeout = Duration::from_secs(self.config.video.job_timeout_secs);
            let poll_start = std::time::Instant::now();

            loop {
                if poll_start.elapsed() > timeout {
                    anyhow::bail!("{operation} job timed out after {}s", timeout.as_secs());
                }

                tokio::time::sleep(Duration::from_secs(3)).await;

                let mut poll_req = self.client.get(&poll_url);
                if let Some(ref api_key) = self.config.video.api_key {
                    poll_req = poll_req.bearer_auth(api_key);
                }

                let resp = poll_req.send().await?.error_for_status()?;
                let body: serde_json::Value = resp.json().await?;

                match body["status"].as_str().unwrap_or("unknown") {
                    "succeeded" | "completed" => {
                        let output_url = body["output"]
                            .as_str()
                            .or_else(|| body["output_url"].as_str())
                            .or_else(|| {
                                body["output"]
                                    .as_array()
                                    .and_then(|a| a.first())
                                    .and_then(|v| v.as_str())
                            })
                            .ok_or_else(|| anyhow::anyhow!("API response missing output URL"))?
                            .to_string();
                        return Ok(output_url);
                    }
                    "failed" | "error" => {
                        let err = body["error"].as_str().unwrap_or("unknown error");
                        anyhow::bail!("{operation} job failed: {err}");
                    }
                    _ => continue,
                }
            }
        }
        .await;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            match result {
                Ok(output_url) => {
                    job.status = JobStatus::Completed;
                    job.output_url = Some(output_url);
                    job.completed_at = Some(now);
                    job.generation_time_ms = Some(elapsed_ms);
                    tracing::info!(job_id = %job_id, operation, elapsed_ms, "{operation} completed");
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.error = Some(e.to_string());
                    job.completed_at = Some(now);
                    job.generation_time_ms = Some(elapsed_ms);
                    tracing::error!(job_id = %job_id, operation, error = %e, "{operation} failed");
                }
            }
        }
    }

    /// Create a lightweight clone for spawning into background tasks.
    fn clone_for_spawn(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: self.client.clone(),
            jobs: self.jobs.clone(),
        }
    }
}

/// Parse "WxH" resolution string into (width, height). Defaults to (768, 768).
fn parse_resolution(resolution: &str) -> (u32, u32) {
    let parts: Vec<&str> = resolution.split('x').collect();
    if parts.len() == 2 {
        let w = parts[0].parse().unwrap_or(768);
        let h = parts[1].parse().unwrap_or(768);
        (w, h)
    } else {
        (768, 768)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_resolution() {
        assert_eq!(parse_resolution("1280x720"), (1280, 720));
        assert_eq!(parse_resolution("512x512"), (512, 512));
        assert_eq!(parse_resolution("invalid"), (768, 768));
        assert_eq!(parse_resolution(""), (768, 768));
    }

    #[test]
    fn test_job_status_serde() {
        let json = serde_json::to_string(&JobStatus::Processing).unwrap();
        assert_eq!(json, "\"processing\"");
        let status: JobStatus = serde_json::from_str("\"completed\"").unwrap();
        assert_eq!(status, JobStatus::Completed);
    }
}
