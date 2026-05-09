#!/bin/bash

echo "Building Docker image..."
docker build -t aml-detection .

echo "Stopping old container if running..."
docker stop aml-detection-app 2>/dev/null || true
docker rm aml-detection-app 2>/dev/null || true

echo "Starting new container..."
docker run -d \
  --name aml-detection-app \
  --env-file .env \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  aml-detection

echo "App is running at http://localhost:8501"