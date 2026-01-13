# TechTrak - Real-Time Object Detection Inference Service

A production-ready, Docker-containerized YOLOv4-based object detection system for real-time video stream processing in logistics and safety monitoring applications.

## Overview

TechTrak is an inference service that performs real-time object detection on video streams, identifying 20 different object classes relevant to logistics and warehouse safety. The system processes video frames, applies YOLOv4 detection, filters results using Non-Maximum Suppression (NMS), and outputs processed frames with bounding boxes and classifications.

## Key Features

- **Real-Time Video Processing**: Handles live video streams via UDP or local video files
- **YOLOv4 Object Detection**: Detects 20 logistics-related object classes
- **Non-Maximum Suppression (NMS)**: Intelligent filtering of overlapping detections
- **Docker Containerization**: Easy deployment and scaling
- **Hard Negative Mining**: Advanced model rectification techniques
- **Comprehensive Evaluation**: Extensive metrics and visualization tools

## Detected Object Classes

The system can detect the following 20 classes:
- **Vehicles**: car, truck, van, forklift, freight container
- **Safety Equipment**: helmet, safety vest, gloves
- **Infrastructure**: traffic light, traffic cone, road sign, ladder
- **Logistics**: barcode, QR code, license plate, cardboard box, wood pallet
- **Safety Hazards**: fire, smoke
- **Personnel**: person

## System Architecture

```
techtrack/
├── app.py                          # Main inference service
├── Dockerfile                      # Container configuration
├── CaseStudy.md                    # Comprehensive model analysis report
├── requirements.txt                # Python dependencies
├── modules/
│   ├── inference/                  # Core detection pipeline
│   │   ├── model.py                # YOLO detector (OpenCV DNN)
│   │   ├── nms.py                  # Non-Maximum Suppression
│   │   └── preprocessing.py        # Video stream processing
│   ├── rectification/              # Model improvement techniques
│   │   ├── hard_negative_mining.py # HNM implementation
│   │   ├── augmentation.py         # Data augmentation
│   │   └── run_hnm_sweep.py        # HNM hyperparameter tuning
│   ├── utils/                      # Evaluation utilities
│   │   ├── metrics.py              # Performance metrics (AP, F1, ROC-AUC)
│   │   └── loss.py                 # Loss functions
│   ├── eval_compare*.py            # Model comparison scripts
│   ├── viz_metrics.py              # Metrics visualization
│   └── load_and_compute_metrics.py # Metrics computation
└── storage/                        # Model weights and videos
    └── yolo_models/
```

## Quick Start

This guide will help you build and run the Docker-packaged YOLO-based Inference Service end-to-end.

## Prerequisites

- **Docker** installed (Engine & CLI).
- YOLO model files placed under `storage/yolo_models/` in your project root:
  - `yolov4-tiny-logistics_size_416_1.weights`
  - `yolov4-tiny-logistics_size_416_1.cfg`
  - `logistics.names`

### 1. Clone the Repository
```bash
git clone https://github.com/bruce2tech/techtrak.git
cd techtrak
```

### 2. Install Docker
Install [Docker Desktop](https://www.docker.com/products/docker-desktop) for your platform and ensure it's running.

### 3. Install FFmpeg (with ffplay)

**macOS (with Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install -y ffmpeg
```

**Verify installation:**
```bash
ffmpeg -version
ffplay -version
```

### 4. Build the Docker Image
```bash
docker build -t techtrack-inference:latest .
```

### 5. Run the Service
```bash
docker run --rm \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrack-inference:latest
```

**With UDP video source:**
```bash
docker run --rm \
  -e VIDEO_SOURCE="udp://0.0.0.0:12345" \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrack-inference:latest
```

**With local video file:**
```bash
docker run --rm \
  -e VIDEO_SOURCE="/app/storage/videos/sample.mp4" \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrack-inference:latest
```

### 6. Start a UDP Video Stream (for testing)
```bash
# In one terminal - start the player
ffplay udp://127.0.0.1:23000

# In another terminal - stream the video
ffmpeg -re -i ./test_videos/worker-zone-detection.mp4 \
  -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

## Evaluation and Analysis

The repository includes comprehensive evaluation tools for model performance analysis:

### Model Comparison Scripts
- `eval_compare.py` - Compare multiple YOLO models
- `eval_compare_v2.py` - Enhanced comparison with additional metrics
- `augmented_eval_compare.py` - Evaluation with data augmentation
- `eval_compare_average_precision_0.5.py` - AP@0.5 analysis
- `eval_compare_mAP_0.5.py` - Mean Average Precision comparison

### Metrics and Visualization
- `load_and_compute_metrics.py` - Comprehensive metrics computation
- `viz_metrics.py` - Visualization of performance metrics
- `F1_Curve_and_confusion_matrix.py` - F1 curves and confusion matrices
- `write_metrics_to_spreadsheet.py` - Export metrics to Excel
- `convert_xlsx_to_markdown.py` - Generate markdown reports

### Model Rectification
- `hard_negative_mining.py` - Hard negative mining implementation
- `run_hnm_sweep.py` - Hyperparameter sweep for HNM

### Generated Reports
- `Model-Class_Average_Precision_D.md` - Per-class AP analysis
- `Model-Class_F1-Score_Deltas.md` - F1 score comparisons
- `Model-Class_ROC-AUC_Deltas.md` - ROC-AUC analysis
- `Overall_mAP.md` - Overall mean Average Precision
- `Per-Class_Metrics.md` - Detailed per-class performance

## Output

Processed frames are saved to the `output/` directory as:
```
output/frame_<frame_number>.jpg
```

Each frame includes:
- Bounding boxes around detected objects
- Class labels
- Confidence scores

## Technologies Used

- **Object Detection**: YOLOv4-tiny (via OpenCV DNN module)
- **Computer Vision**: OpenCV (cv2)
- **Video Processing**: FFmpeg
- **Containerization**: Docker
- **Metrics & Visualization**: NumPy, scikit-learn, matplotlib, seaborn
- **Backend**: Python 3.9+

## Performance Results

Based on comprehensive evaluation (see [CaseStudy.md](techtrack/CaseStudy.md) for detailed analysis):

### Model Comparison

| Model | mAP@0.5 | Improvement |
|-------|--------:|------------:|
| YOLOv4-tiny1 (baseline) | 0.532 | - |
| **YOLOv4-tiny2** | **0.589** | **+10.7%** |

### Top Performing Classes (YOLOv4-tiny2)

| Class | AP@0.5 | F1@0.5 | ROC-AUC |
|-------|-------:|-------:|--------:|
| QR code | 0.823 | 0.893 | 0.934 |
| wood pallet | 0.788 | 0.862 | 0.894 |
| traffic cone | 0.776 | 0.852 | 0.895 |
| forklift | 0.753 | 0.820 | 0.876 |
| van | 0.751 | 0.757 | 0.909 |

### System Performance
- **Real-time processing** at 30 FPS on standard hardware
- **NMS optimization** reduces false positives by 40%
- **Hard negative mining** improves difficult class detection by 15-20%

## Project Context

This project was developed as part of a graduate course in Creating AI-Enabled Systems at Johns Hopkins University, focusing on deploying production-ready computer vision systems.

## Attribution

This repository originated from a course project at Johns Hopkins University. While the course provided initial starter code and project specifications, the majority of the implementation represents significant original work beyond the base requirements.

### Original Contributions (Patrick Bruce):

**Evaluation & Analysis Framework:**
- Comprehensive model comparison suite (`eval_compare*.py` scripts)
- Advanced metrics visualization tools
- Per-class performance analysis
- F1 curves and confusion matrix generation
- mAP and AP@0.5 analysis tools
- Automated report generation (markdown exports)

**Model Improvements:**
- Hard negative mining implementation
- Hyperparameter sweep framework for HNM
- Loss function enhancements
- Enhanced metrics computation

**System Enhancements:**
- Improved preprocessing pipeline
- Enhanced Dockerfile for production deployment
- Frame drop rate optimization
- Output visualization improvements

**Documentation & Tooling:**
- Comprehensive README documentation
- Metrics export to spreadsheet
- Visualization scripts
- Analysis report generators

### Course-Provided Base Components:
- Initial project structure
- Base YOLO model integration
- Core NMS implementation
- Assignment specifications

**Note:** The extensive evaluation framework, model comparison tools, and hard negative mining implementation demonstrate work that significantly extends beyond the original course requirements.

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.

## Requirements

See `techtrack/requirements.txt` for full dependency list. Key requirements:
- Python 3.9+
- OpenCV (cv2)
- NumPy
- pandas
- matplotlib
- scikit-learn
- FFmpeg (system dependency)