FROM python:3.10

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# DeepTag expects models in a 'models/' directory relative to the CWD
RUN ln -s deeptag/models models

# Expose the Flask web server port
EXPOSE 5000

CMD ["python", "app.py"]
