# Public NVIDIA Nemotron OCR v2 on Nostromo

This standalone service runs NVIDIA's published OCR pipeline on the RTX 4090
(24 GiB). It uses the public PyTorch image and Hugging Face model; an NVIDIA
enterprise subscription and NGC key are unnecessary. Upstream code and English
weights are pinned to revision `0e83e83f17943524b90afa6c0fd82ac2bc1a40ca`.
The CUDA extension is compiled for Ada compute capability 8.9.

`server.py` only translates images and raw detections into the NIM 2.x wire
format. It does not reconstruct document flow, paragraphs, or tables and does
not import code from `nemotron-ocr-spark`. i2klib owns that processing in Java
and produces its existing canonical i2k Markdown v2 output.

## Run

Docker and the NVIDIA Container Toolkit's CDI device must already work:

```bash
cd /home/josh/i2k/projects/nemotron-ocr-public
./run.sh build
./run.sh start
./run.sh status
./run.sh logs
./run.sh stop
```

The default listener is `127.0.0.1:8080`. `PORT` overrides the host port;
`OCR_CACHE` overrides the default `cache/` directory. The model cache is mounted
writable for the invoking user. First startup downloads pinned English weights
from Hugging Face; later starts reuse them. Nostromo's existing downloaded
weights were copied into this independent cache during migration.

The script creates container `nemotron-ocr-upstream`. To recreate it after an
image or port change, stop and remove that container, then run `start` again.
The container runs as the invoking user with one GPU and 2 GiB shared memory.
English recognition uses upstream detector length 1536; inference is serialized.

## API

- `GET /v1/health/ready`: returns readiness after model initialization.
- `POST /v1/ocr`: JSON `input` array with objects containing
  `type: "image_url"` and `url: "data:image/png;base64,..."` (JPEG also accepted).
  `merge_levels: ["word"]` requests raw words.
- Responses contain `data[].index` and `data[].text_detections[]` with normalized
  four-point `bounding_box.points` and `text_prediction.text/confidence`.

This adapter implements the subset used by i2klib, rather than a full NIM
replacement: data URLs only, one merge level per request, up to 16 images,
32 MiB request bodies, and 8 million pixels per raster. i2klib tiles larger
images. Upstream provides axis-aligned bounds; the adapter preserves this
geometry and uses confidence 0.5 when upstream supplies no confidence.

## Java integration and verification

Build i2klib core and the existing `i2klib-ocr-nemotron` add-on with Java 17.
Use `new NemotronOcrEngine(URI.create("http://127.0.0.1:8080"))` and existing
`ParserOptions.withOcrFallback(engine)`. See the component's `OCR_ADDONS.md`.

```bash
cd /home/josh/i2k/i2ksuite/i2klib
export JAVA_CMD=/usr/lib/jvm/java-17-openjdk/bin/java
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
mvn -f addons/ocr-nemotron/pom.xml \
  -Dnemotron.url=http://127.0.0.1:8080 \
  -Dnemotron.output=/tmp/nemotron-java-verification test
```

The live test verifies synthetic images and scanned PDFs, reconstructed tables,
canonical Markdown round trips, and sideways recognition. This is an integration
check, not an accuracy benchmark on the production corpus.

The former `nemotron-ocr-public` container was stopped and retained for rollback.
To roll back, stop this service and start that retained container. Do not run
both on port 8080. The Eric checkout remains available for comparison.

Upstream implementation: Apache-2.0. Model weights: NVIDIA Open Model License
Agreement. The base image carries its published NVIDIA container terms.
