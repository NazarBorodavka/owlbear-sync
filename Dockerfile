FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install minimal dependencies for AprilTag-only runtime/build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY tracker/requirements.txt ./tracker/

# Upgrade pip, setuptools, wheel first
RUN pip install --no-cache-dir --break-system-packages --upgrade pip setuptools wheel 2>&1 | tail -20

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r tracker/requirements.txt 2>&1 | tail -50

# Verify cv2 is importable
RUN python -c "import cv2; print(f'[OK] cv2 version: {cv2.__version__}')" || (echo "[ERROR] cv2 import failed" && exit 1)

COPY tracker ./tracker
COPY extension ./extension

# Expose the Flask web server port
EXPOSE 5000

WORKDIR /app/tracker
CMD ["python", "app.py"]

