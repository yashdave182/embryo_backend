FROM python:3.10-slim

# libgl1-mesa-glx was renamed to libgl1 in Debian Trixie
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HuggingFace requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py .
COPY --chown=user efficientnet_embryo_model.h5 .
COPY --chown=user dual_branch_embryo_model.keras .
COPY --chown=user morph_scaler.pkl .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
