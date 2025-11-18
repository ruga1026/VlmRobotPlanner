from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="/tmp/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False,
    revision="main",
)
print("saved to /tmp/all-MiniLM-L6-v2")