FROM nvcr.io/nvidia/pytorch:25.11-py3
ARG NEMOTRON_REVISION=0e83e83f17943524b90afa6c0fd82ac2bc1a40ca
ENV NEMOTRON_REVISION=${NEMOTRON_REVISION}
ARG TORCH_CUDA_ARCH_LIST=8.9
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FORCE_CUDA=1 MAX_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir setuptools wheel ninja pybind11 hatchling fastapi uvicorn pillow
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nvidia/nemotron-ocr-v2 /opt/nemotron \
    && cd /opt/nemotron && GIT_LFS_SKIP_SMUDGE=1 git checkout "$NEMOTRON_REVISION" \
    && python -m pip install --no-cache-dir --no-build-isolation ./nemotron-ocr
COPY server.py /opt/server.py
ENV HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch NEMOTRON_LANG=en NEMOTRON_INFER_LENGTH=1536
WORKDIR /opt
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
