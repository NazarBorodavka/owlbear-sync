FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies for CCTag runtime and build, plus a real system
# ffmpeg + Intel VAAPI stack for optional hardware-accelerated RTSP decode
# (see IPCameraCaptureVAAPI in tracker/app.py). This is unrelated to the
# FFmpeg bundled inside the opencv-python-headless wheel that cv2.VideoCapture
# uses by default — that bundled build is a generic binary with no VAAPI
# support compiled in, which is a hard limitation of the PyPI wheel, not
# something togglable at runtime. Debian's own ffmpeg package *does* support
# VAAPI, so the hardware-accel path shells out to it directly instead.
# These packages are always installed (small, harmless if unused) — actually
# using hardware decode still requires opting in via TRACKER_USE_VAAPI and
# passing /dev/dri through in docker-compose.yml, since not every host has
# an Intel iGPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libboost-all-dev \
    libtbb-dev \
    libeigen3-dev \
    ffmpeg \
    vainfo \
    intel-media-va-driver \
    libva2 \
    libva-drm2 \
    && rm -rf /var/lib/apt/lists/*

COPY tracker/requirements.txt ./tracker/

# Upgrade pip, setuptools, wheel first
RUN pip install --no-cache-dir --break-system-packages --upgrade pip setuptools wheel 2>&1 | tail -20

# Install Python dependencies from requirements.txt (including pybind11)
RUN pip install --no-cache-dir --break-system-packages -r tracker/requirements.txt pybind11 2>&1 | tail -50

# Build CCTag C++ library (CPU-only: full-frame detection on 1080p takes
# ~300ms/frame on CPU, which is why the recognizer runs on a background
# thread decoupled from the 30fps disk tracker in tracker/app.py. If you
# deploy on a host with an NVIDIA GPU + nvidia-container-toolkit, switching
# this to a `nvidia/cuda:*-devel` base image and -DCCTAG_WITH_CUDA=ON can
# reportedly give an order-of-magnitude speedup, letting recognition run
# near every frame instead of riding on stale votes between detections.
COPY CCTag-develop ./CCTag-develop
RUN mkdir -p CCTag-develop/build && cd CCTag-develop/build && \
    cmake .. -DCCTAG_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j2 && make install && ldconfig

# Build python wrapper
COPY python ./python
RUN cd python && python setup_cctag.py build_ext --inplace

# Verify cv2 is importable
RUN python -c "import cv2; print(f'[OK] cv2 version: {cv2.__version__}')" || (echo "[ERROR] cv2 import failed" && exit 1)

COPY tracker ./tracker
COPY extension ./extension
COPY tags ./tags

# Build identity, baked in by CI so the dashboard can show which image is
# actually running. Declared this late (after the expensive apt/pip/CCTag
# compile steps above) on purpose: these values change on every CI run, and
# Docker's layer cache invalidates every layer from the first changed one
# onward — putting them earlier would force a full CCTag rebuild every push.
ARG BUILD_VERSION=dev
ARG BUILD_BRANCH=local
ARG BUILD_COMMIT=unknown
ENV BUILD_VERSION=$BUILD_VERSION
ENV BUILD_BRANCH=$BUILD_BRANCH
ENV BUILD_COMMIT=$BUILD_COMMIT

# Expose the Flask web server port
EXPOSE 5000

WORKDIR /app/tracker
CMD ["python", "app.py"]

