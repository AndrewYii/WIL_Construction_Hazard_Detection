FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio \
    && pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY runs/detect/plan_a_yolov8/weights/best.pt ./runs/detect/plan_a_yolov8/weights/best.pt

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
