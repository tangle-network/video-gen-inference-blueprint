use video_gen_inference::video::{
    Img2VidRequest, InterpolateRequest, JobKind, JobStatus, UpscaleRequest,
    VideoGenRequest, VideoJob,
};

#[test]
fn job_status_serialization_roundtrip() {
    for (status, expected_str) in [
        (JobStatus::Queued, "\"queued\""),
        (JobStatus::Processing, "\"processing\""),
        (JobStatus::Completed, "\"completed\""),
        (JobStatus::Failed, "\"failed\""),
    ] {
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, expected_str);

        let deserialized: JobStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }
}

#[test]
fn job_kind_serialization() {
    for (kind, expected) in [
        (JobKind::Generate, "\"generate\""),
        (JobKind::Img2Vid, "\"img2vid\""),
        (JobKind::Upscale, "\"upscale\""),
        (JobKind::Interpolate, "\"interpolate\""),
    ] {
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, expected);
    }
}

#[test]
fn video_gen_request_serialization() {
    let req = VideoGenRequest {
        prompt: "a sunset over mountains".to_string(),
        duration_secs: 5,
        resolution: "1280x720".to_string(),
        fps: 24,
    };

    let json = serde_json::to_string(&req).unwrap();
    let deserialized: VideoGenRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.prompt, "a sunset over mountains");
    assert_eq!(deserialized.duration_secs, 5);
    assert_eq!(deserialized.resolution, "1280x720");
    assert_eq!(deserialized.fps, 24);
}

#[test]
fn video_job_completed_state() {
    let job = VideoJob {
        job_id: "test-123".to_string(),
        status: JobStatus::Completed,
        prompt: "test prompt".to_string(),
        duration_secs: 10,
        resolution: "1920x1080".to_string(),
        fps: 30,
        output_url: Some("https://cdn.example.com/video.mp4".to_string()),
        error: None,
        created_at: 1700000000,
        completed_at: Some(1700000060),
        generation_time_ms: Some(58000),
    };

    let json = serde_json::to_string(&job).unwrap();
    let deserialized: VideoJob = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.job_id, "test-123");
    assert_eq!(deserialized.status, JobStatus::Completed);
    assert_eq!(
        deserialized.output_url.as_deref(),
        Some("https://cdn.example.com/video.mp4")
    );
    assert!(deserialized.error.is_none());
    assert_eq!(deserialized.generation_time_ms, Some(58000));
}

#[test]
fn video_job_failed_state() {
    let job = VideoJob {
        job_id: "fail-456".to_string(),
        status: JobStatus::Failed,
        prompt: "impossible prompt".to_string(),
        duration_secs: 120,
        resolution: "7680x4320".to_string(),
        fps: 60,
        output_url: None,
        error: Some("OOM: insufficient VRAM".to_string()),
        created_at: 1700000000,
        completed_at: Some(1700000005),
        generation_time_ms: Some(5000),
    };

    let json = serde_json::to_string(&job).unwrap();
    let deserialized: VideoJob = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.status, JobStatus::Failed);
    assert!(deserialized.output_url.is_none());
    assert_eq!(
        deserialized.error.as_deref(),
        Some("OOM: insufficient VRAM")
    );
}

#[test]
fn img2vid_request_serialization() {
    let req = Img2VidRequest {
        image_bytes: vec![0x89, 0x50, 0x4E, 0x47], // PNG header bytes
        prompt: "animate this image".to_string(),
        duration_secs: 3,
        params: serde_json::json!({"motion_strength": 0.8}),
    };

    let json = serde_json::to_string(&req).unwrap();
    let deserialized: Img2VidRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.image_bytes, vec![0x89, 0x50, 0x4E, 0x47]);
    assert_eq!(deserialized.prompt, "animate this image");
    assert_eq!(deserialized.duration_secs, 3);
}

#[test]
fn upscale_request_serialization() {
    let req = UpscaleRequest {
        video_url: "https://cdn.example.com/low_res.mp4".to_string(),
        target_resolution: "3840x2160".to_string(),
    };

    let json = serde_json::to_string(&req).unwrap();
    let deserialized: UpscaleRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.target_resolution, "3840x2160");
}

#[test]
fn interpolate_request_serialization() {
    let req = InterpolateRequest {
        video_url: "https://cdn.example.com/video.mp4".to_string(),
        target_fps: 60,
    };

    let json = serde_json::to_string(&req).unwrap();
    let deserialized: InterpolateRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.target_fps, 60);
}
