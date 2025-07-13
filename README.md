# TechTrack Case Study Base Repository

This is the repository for TechTrack. This contains the resources (e.g., instructions, unit tests, and boiler plate code) to complete the assignments and analysis. You can find the instructions using the following links:

- [Assignment 2](assignment-2-test): Implmentation
- [Assignment 3](assignment-3-test): Implementation
- [Assignment 4](assignment-4-test): Design and Analysis

# Quick Start: TechTrack Inference Service

This guide will help you build and run the Docker-packaged YOLO-based Inference Service end-to-end.

## Prerequisites

- **Docker** installed (Engine & CLI).
- YOLO model files placed under `storage/yolo_models/` in your project root:
  - `yolov4-tiny-logistics_size_416_1.weights`
  - `yolov4-tiny-logistics_size_416_1.cfg`
  - `logistics.names`

# 1. Clone the Repository
    '''
    bash
    git clone <your-repo-url>
    cd project-techtrak-bruce2tech
    '''
# 2. Install Docker
    Install [Docker Desktop](https://www.docker.com/products/docker-desktop) for your platform and ensure it’s running.

# 2. FFmpeg (with ffplay)
   - **macOS (with Homebrew):**  
     ```bash
     brew install ffmpeg
     ```  
     This provides `ffmpeg`, `ffprobe`, and `ffplay`.  
   - **Linux (Debian/Ubuntu):**  
     ```bash
     sudo apt update
     sudo apt install -y ffmpeg
     ```  
   - **Verify installation:**  
     ```bash
     ffmpeg -version
     ffplay -version
     ```



# 4. Build the Docker Image
# From the project root (where your Dockerfile lives), build:
docker build -t techtrack-inference:latest .


# 4. Run the service
docker run --rm \
  -v $(pwd)/
  storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrack-inference:latest



docker run --rm \
  -e VIDEO_SOURCE="udp://0.0.0.0:12345" \
# Or for a local MP4:
# e VIDEO_SOURCE="/app/storage/videos/sample.mp4" \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrack-inference:latest


output/frame_<frame_number>.jpg

# 5. Start your UDP video stream
ffplay udp://127.0.0.1:23000
ffmpeg -re -i ./test_videos/worker-zone-detection.mp4 \
  -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000