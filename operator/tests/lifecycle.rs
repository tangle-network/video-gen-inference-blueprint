//! Full lifecycle test -- video gen submission through real handler + wiremock backend.

use anyhow::{Result, ensure};
use wiremock::{MockServer, Mock, ResponseTemplate, matchers::{method, path}};
use video_gen_inference::VideoGenJobRequest;

#[tokio::test]
async fn test_submit_video_direct_with_wiremock() -> Result<()> {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/video/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "job_id": "vid-job-abc123",
            "status": "queued"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    video_gen_inference::init_direct_for_testing(&mock_server.uri());

    let request = VideoGenJobRequest {
        prompt: "A cat playing piano".into(),
        durationSecs: 5,
        resolution: "1280x720".into(),
        fps: 24,
    };

    let result = video_gen_inference::submit_video_direct(&request).await;

    match result {
        Ok(job_id) => {
            ensure!(
                job_id == "vid-job-abc123",
                "expected job_id 'vid-job-abc123', got '{job_id}'"
            );
        }
        Err(e) => panic!("Video gen submission failed: {e}"),
    }

    mock_server.verify().await;

    Ok(())
}
