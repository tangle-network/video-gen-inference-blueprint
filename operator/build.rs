use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=src");

    let blueprint_metadata = serde_json::json!({
        "name": "video-gen-inference",
        "description": "Video generation and image synthesis operator via ComfyUI workflows on Tangle",
        "version": env!("CARGO_PKG_VERSION"),
        "manager": {
            "Evm": "VideoGenBSM"
        },
        "master_revision": "Latest",
        "jobs": [
            {
                "name": "generate_video",
                "job_index": 0,
                "description": "Generate video or image from text/image prompt via ComfyUI",
                "inputs": ["(string,string,uint32,uint32,uint32,uint32)"],
                "outputs": ["(bytes,string,uint32,uint32,uint32)"],
                "required_results": 1,
                "execution": "local"
            }
        ]
    });

    let json = serde_json::to_string_pretty(&blueprint_metadata).unwrap();
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().expect("workspace root");
    std::fs::write(workspace_root.join("blueprint.json"), json.as_bytes()).unwrap();
}
