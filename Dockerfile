FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /code

# Organize training_code/ per ZAC 2026 Docker Guide without duplicating local repo files
RUN mkdir -p /code/training_code && \
    cp /code/train_sft.py /code/train_grpo_rlvr.py /code/merge_whisper_lora.py /code/evaluate.py /code/training_code/ && \
    cp -r /code/configs /code/scripts /code/src /code/training_code/ && \
    sed -i 's/\r$//' /code/predict.sh /code/start_jupyter.sh /code/entrypoint.sh && \
    chmod +x /code/predict.sh /code/start_jupyter.sh /code/entrypoint.sh

EXPOSE 9777

CMD ["/bin/bash", "/code/predict.sh"]
