# Use an official NVIDIA CUDA image as a base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV REACT_VERSION=18.2.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your application code to the container
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Expose the port on which your Dash app will run
EXPOSE 8050

# Run your application
CMD ["python3", "app.py"]