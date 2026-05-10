FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install system dependencies for building stag-python and running OpenCV
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the Flask web server port
EXPOSE 5000

CMD ["python", "app.py"]
