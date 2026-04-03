use std::sync::Arc;

use tokio::sync::Semaphore;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use video_gen_inference::config::{
    BillingConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig, VideoBackendMode,
    VideoConfig,
};
use video_gen_inference::video::VideoBackend;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn test_config(backend_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                .into(),
            tangle_core: "0x0000000000000000000000000000000000000000".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        video: VideoConfig {
            mode: VideoBackendMode::Api,
            endpoint: format!("http://127.0.0.1:{backend_port}"),
            api_key: None,
            model: "test-video-model".into(),
            max_duration_secs: 10,
            default_fps: 24,
            supported_resolutions: vec![
                "512x512".into(),
                "768x768".into(),
                "1280x720".into(),
            ],
            default_resolution: "768x768".into(),
            job_timeout_secs: 30,
            comfyui_workflow: None,
            output_dir: std::path::PathBuf::from("/tmp/test-videos"),
            startup_timeout_secs: 5,
            supported_operations: vec![
                "generate".into(),
                "img2vid".into(),
                "upscale".into(),
                "interpolate".into(),
            ],
            max_input_size_bytes: 64 * 1024 * 1024,
        },
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            max_concurrent_jobs: 4,
            max_request_body_bytes: 16 * 1024 * 1024,
            max_per_account_jobs: 0,
        },
        billing: BillingConfig {
            required: false,
            price_per_second: 100000,
            max_spend_per_request: 10000000,
            min_credit_balance: 1000,
            billing_required: false,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            gpu_model: None,
            monitor_interval_secs: 30,
        },
        qos: None,
    }
}

async fn start_test_server(
    backend_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(backend_port);
    config.server.port = server_port;
    let config = Arc::new(config);

    let backend = Arc::new(VideoBackend::new(config.clone()).unwrap());
    let semaphore = Arc::new(Semaphore::new(4));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let state = video_gen_inference::server::AppState {
        config,
        backend,
        semaphore,
    };

    let handle = video_gen_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

// -- Tests --

#[tokio::test]
async fn test_health_check_healthy() {
    let mock = MockServer::start().await;

    // API mode uses /health for health checks
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model"], "test-video-model");
}

#[tokio::test]
async fn test_health_check_unhealthy() {
    let mock = MockServer::start().await;
    // no mock -> 404 -> unhealthy

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn test_submit_video_job_and_poll() {
    let mock = MockServer::start().await;

    // Health check for backend
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    // API mode: POST to root returns job id, the backend spawns async processing.
    // The video backend's execute_api POSTs to the endpoint root.
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "remote-job-123"
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();

    // Submit a job
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/video/generate"))
        .json(&serde_json::json!({
            "prompt": "a sunset over mountains",
            "duration_secs": 4,
            "resolution": "768x768",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["job_id"].as_str().is_some());
    assert_eq!(body["status"], "queued");
    let job_id = body["job_id"].as_str().unwrap();

    // Poll for the job (it will be queued or processing)
    let resp = client
        .get(format!("http://127.0.0.1:{port}/v1/video/{job_id}"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["job_id"], job_id);
    // Status should be queued or processing (async execution)
    let status = body["status"].as_str().unwrap();
    assert!(
        status == "queued" || status == "processing" || status == "completed" || status == "failed",
        "unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_submit_video_empty_prompt() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/video/generate"))
        .json(&serde_json::json!({
            "prompt": "",
            "duration_secs": 4,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("empty"));
}

#[tokio::test]
async fn test_submit_video_duration_exceeded() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/video/generate"))
        .json(&serde_json::json!({
            "prompt": "test",
            "duration_secs": 999,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("exceeds"));
}

#[tokio::test]
async fn test_submit_video_unsupported_resolution() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/video/generate"))
        .json(&serde_json::json!({
            "prompt": "test",
            "duration_secs": 4,
            "resolution": "9999x9999",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("not supported"));
}

#[tokio::test]
async fn test_get_nonexistent_job() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/v1/video/nonexistent-id"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("not found"));
}
