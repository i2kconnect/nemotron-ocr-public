#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
name=nemotron-ocr-upstream
image=nemotron-ocr-upstream:public-v2
port=${PORT:-8080}
cache=${OCR_CACHE:-$root/cache}
case ${1:-start} in
    build) exec docker build -t "$image" "$root" ;;
    status) docker ps -a --filter "name=^/${name}$" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'; exec curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/v1/health/ready" ;;
    logs) exec docker logs --tail 100 --follow "$name" ;;
    stop) exec docker stop "$name" ;;
    start) ;;
    *) echo "Usage: $0 {build|start|status|logs|stop}" >&2; exit 2 ;;
esac
mkdir -p "$cache"
docker run --detach --pull=never --name "$name" \
    --device nvidia.com/gpu=0 --shm-size=2g --user "$(id -u):$(id -g)" \
    --publish "127.0.0.1:$port:8000" \
    --mount "type=bind,source=$cache,target=/cache" \
    --env HF_HOME=/cache --env TORCH_HOME=/cache/torch \
    --env NEMOTRON_LANG=en --env NEMOTRON_INFER_LENGTH=1536 "$image"
