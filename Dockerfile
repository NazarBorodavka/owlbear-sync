FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies for CCTag runtime and build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libboost-all-dev \
    libtbb-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY tracker/requirements.txt ./tracker/

# Upgrade pip, setuptools, wheel first
RUN pip install --no-cache-dir --break-system-packages --upgrade pip setuptools wheel 2>&1 | tail -20

# Install Python dependencies from requirements.txt (including pybind11)
RUN pip install --no-cache-dir --break-system-packages -r tracker/requirements.txt pybind11 2>&1 | tail -50

# Build CCTag C++ library
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

# Expose the Flask web server port
EXPOSE 5000

WORKDIR /app/tracker
CMD ["python", "app.py"]

