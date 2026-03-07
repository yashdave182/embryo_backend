FROM python:3.10-slim

# System deps for OpenCV (libgl1-mesa-glx renamed to libgl1 in Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps as ROOT so they go to system site-packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app and model files
COPY app.py .
COPY efficientnet_embryo_model.h5 .
COPY dual_branch_embryo_model.keras .
COPY morph_scaler.pkl .

# Switch to non-root user AFTER installing deps (Render requirement)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
