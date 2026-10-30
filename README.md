# Nemotron OCR v2 HTTP service

A small HTTP adapter for [NVIDIA Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2).
It runs the upstream Python OCR pipeline on an NVIDIA GPU and exposes raw text
detections through a subset of the NIM 2.x OCR API. Clients can use the results
for their own document layout, paragraph, and table reconstruction.

The image installs NVIDIA's Python implementation and downloads English model
weights from Hugging Face. Both are pinned to revision
`0e83e83f17943524b90afa6c0fd82ac2bc1a40ca`.

## Requirements

- Docker and NVIDIA Container Toolkit with CDI configured. The helper script
  selects the device `nvidia.com/gpu=0`.
- An NVIDIA GPU supported by the upstream model and the container's CUDA/PyTorch
  stack, with sufficient device memory for inference.
- A host NVIDIA driver compatible with the `nvcr.io/nvidia/pytorch:25.11-py3`
  base image.
- Network access to obtain the container image, Python dependencies, upstream
  source, and model weights. Subsequent starts reuse cached weights.

## Build

```bash
git clone https://github.com/i2kconnect/nemotron-ocr-public.git
cd nemotron-ocr-public
```

The upstream CUDA extension must be compiled for the target GPU's compute
capability. The Dockerfile defaults to `8.9`; select a different target with
the `TORCH_CUDA_ARCH_LIST` build argument:

```bash
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="<compute-capability>" \
  -t nemotron-ocr-upstream:public-v2 .
```

Replace `<compute-capability>` with the target GPU's value. Multiple targets
can be supplied as a quoted, semicolon-separated list, provided the container's
CUDA toolchain supports them. For the default target, use `./run.sh build`.
Selecting a build target alone does not establish model compatibility or memory
requirements for a particular GPU.

## Run

```bash
./run.sh start
./run.sh status
./run.sh logs
./run.sh stop
```

The service listens at `http://127.0.0.1:8080`. The helper creates a container
named `nemotron-ocr-upstream`, runs it as the invoking user with 2 GiB of shared
memory, and mounts a writable model cache. First startup downloads the weights
and initializes the model; wait for readiness before sending OCR requests.

| Setting | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | Host port, bound to loopback |
| `OCR_CACHE` | `cache/` inside the checkout | Host directory for downloaded weights |

For example:

```bash
PORT=8090 OCR_CACHE=/absolute/path/to/model-cache ./run.sh start
PORT=8090 ./run.sh status
```

The helper's `start` command creates a new container. After stopping it, remove
the existing container before starting again or changing its image or settings:

```bash
./run.sh stop
docker rm nemotron-ocr-upstream
./run.sh start
```

The bind-mounted model cache remains available after container removal. The
shipped configuration loads English weights, uses detector length `1536`, and
serializes inference. Other languages require corresponding model downloads
and server configuration; changing a language environment variable alone is
insufficient.

## API

### Readiness

```bash
curl --fail http://127.0.0.1:8080/v1/health/ready
```

### Recognize an image

Send `POST /v1/ocr` with a base64 PNG or JPEG data URL:

```json
{
  "input": [
    {"type": "image_url", "url": "data:image/png;base64,<base64-image-bytes>"}
  ],
  "merge_levels": ["word"]
}
```

For example, using Python's standard library to create the request:

```bash
python3 - <<'PY' > request.json
import base64
import json
from pathlib import Path

encoded = base64.b64encode(Path("document.png").read_bytes()).decode("ascii")
print(json.dumps({
    "input": [{"type": "image_url", "url": "data:image/png;base64," + encoded}],
    "merge_levels": ["word"],
}))
PY
curl --fail-with-body http://127.0.0.1:8080/v1/ocr \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

Responses contain `data[].index` and `data[].text_detections[]`. Each detection
includes a normalized four-point `bounding_box.points` polygon and
`text_prediction.text` and `text_prediction.confidence` fields.

The adapter supports:

- PNG and JPEG data URLs; remote image URLs and direct PDF input are unsupported.
- One merge level per request: `word` (default), `sentence`, or `paragraph`.
- Between 1 and 16 images per request.
- A maximum JSON request body of 32 MiB and 8 million pixels per image.

Clients must rasterize PDFs and tile larger images before submitting them.
The adapter preserves upstream axis-aligned bounds and uses confidence `0.5`
when upstream supplies no confidence. It does not reconstruct reading order,
tables, or formatted documents and does not implement the complete NIM API.

## Licensing

The upstream implementation is Apache-2.0 licensed. Model weights use the
NVIDIA Open Model License Agreement. See the
[upstream repository](https://huggingface.co/nvidia/nemotron-ocr-v2) for its
license files and model terms. The base image is subject to its published
NVIDIA container terms.
