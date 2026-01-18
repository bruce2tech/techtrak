# TechTrak: Real-Time Object Detection for Logistics Safety

> Docker-containerized YOLOv4 inference service achieving 0.589 mAP@0.5 on 20 logistics-specific classes, with analysis of hard negative mining for improving detection of underperforming categories.

## The Problem

Warehouse and logistics environments present distinct object detection requirements:

1. **Domain-specific classes**: Generic COCO-trained detectors miss logistics-critical objects (forklifts, pallets, QR codes, safety equipment)
2. **Real-time constraints**: Video stream processing must maintain 30 FPS for safety monitoring
3. **Class imbalance**: Some objects (forklifts, fire) appear rarely but are high-priority detections
4. **Overlapping detections**: Dense environments produce redundant bounding boxes requiring intelligent suppression

This project addresses these through domain-specific model fine-tuning, optimized NMS, and hard negative mining for difficult classes.

## Key Findings

### Model Improvement Through Fine-Tuning

I evaluated baseline and fine-tuned YOLOv4-tiny models on the logistics dataset:

| Model | mAP@0.5 | Improvement |
|-------|--------:|------------:|
| YOLOv4-tiny1 (baseline) | 0.532 | — |
| **YOLOv4-tiny2 (fine-tuned)** | **0.589** | **+10.7%** |

### Per-Class Performance Analysis

The fine-tuned model shows significant variation across classes, revealing where additional work is needed:

**High-Performing Classes** (AP@0.5 > 0.75):
| Class | AP@0.5 | F1@0.5 | ROC-AUC |
|-------|-------:|-------:|--------:|
| QR code | 0.823 | 0.893 | 0.934 |
| wood pallet | 0.788 | 0.862 | 0.894 |
| traffic cone | 0.776 | 0.852 | 0.895 |
| forklift | 0.753 | 0.820 | 0.876 |

**Classes Requiring Improvement** (AP@0.5 < 0.50):
| Class | AP@0.5 | Primary Issue |
|-------|-------:|---------------|
| fire | 0.312 | Rare occurrence, visual variability |
| gloves | 0.389 | Small object, occlusion |
| smoke | 0.421 | Amorphous boundaries |

### Hard Negative Mining Impact

I implemented hard negative mining (HNM) to address false positive patterns. The technique mines high-confidence incorrect predictions and re-weights them during training:

- **False positive reduction**: 40% decrease in spurious detections
- **Difficult class improvement**: 15-20% AP gain on underperforming classes
- **Trade-off observed**: Slight regression on already-strong classes (1-2% AP)

**Strategic Decision**: HNM is most effective when applied selectively to weak classes rather than globally. Per-class confidence thresholds outperform uniform thresholds.

## System Architecture

```
techtrak/
├── app.py                      # Main inference service
├── Dockerfile                  # Container configuration
├── CaseStudy.md                # Comprehensive model analysis
├── modules/
│   ├── inference/              # Core detection pipeline
│   │   ├── model.py            # YOLO detector (OpenCV DNN)
│   │   ├── nms.py              # Non-Maximum Suppression
│   │   └── preprocessing.py    # Video stream processing
│   ├── rectification/          # Model improvement
│   │   ├── hard_negative_mining.py
│   │   ├── augmentation.py
│   │   └── run_hnm_sweep.py    # HNM hyperparameter tuning
│   └── utils/                  # Evaluation tools
│       ├── metrics.py          # AP, F1, ROC-AUC computation
│       └── loss.py             # Loss functions
├── storage/
│   ├── videos/                 # Input video files (.mp4)
│   └── yolo_models/            # Model weights and configs
└── output/                     # Processed frames with detections
```

> **Note**: Model weights and sample videos are available in the [GitHub Release](https://github.com/bruce2tech/techtrak/releases).

## Detected Object Classes

The system detects 20 logistics-relevant classes:

| Category | Classes |
|----------|---------|
| **Vehicles** | car, truck, van, forklift, freight container |
| **Safety Equipment** | helmet, safety vest, gloves |
| **Infrastructure** | traffic light, traffic cone, road sign, ladder |
| **Logistics** | barcode, QR code, license plate, cardboard box, wood pallet |
| **Safety Hazards** | fire, smoke |
| **Personnel** | person |

## Quick Start

### Prerequisites
- Docker
- FFmpeg (for video streaming tests)
- YOLO model files in `storage/yolo_models/`:
  - `yolov4-tiny-logistics_size_416_1.weights`
  - `yolov4-tiny-logistics_size_416_1.cfg`
  - `logistics.names`
- Download the YOLO model files and video https://github.com/bruce2tech/techtrak/releases/tag/Model_and_videos
### Build and Run

```bash
git clone https://github.com/bruce2tech/techtrak.git
cd techtrak

# Build Docker image
docker build -t techtrak-inference:latest .

# Run with local video file
docker run --rm \
  -e VIDEO_SOURCE="/app/storage/videos/Safety_Full_Hat_and_Vest.mp4" \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrak-inference:latest

# Run with UDP video stream
docker run --rm \
  -e VIDEO_SOURCE="udp://0.0.0.0:12345" \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/output:/app/output \
  techtrak-inference:latest
```

### Testing with Video Stream

```bash
# Terminal 1: Start video player
ffplay udp://127.0.0.1:23000

# Terminal 2: Stream test video
ffmpeg -re -i ./storage/videos/sample.mp4 \
  -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```
Note: This will stream the video and save the detections, but does not display detections while streaming.
## Output

Processed frames are saved to `output/` with:
- Bounding boxes around detected objects
- Class labels
- Confidence scores

```
output/frame_0001.jpg
output/frame_0002.jpg
...
```

## Evaluation Suite

The repository includes comprehensive evaluation tools for model analysis:

| Script | Purpose |
|--------|---------|
| `eval_compare.py` | Multi-model comparison |
| `eval_compare_mAP_0.5.py` | Mean Average Precision analysis |
| `F1_Curve_and_confusion_matrix.py` | Per-class F1 and confusion matrices |
| `run_hnm_sweep.py` | Hard negative mining hyperparameter optimization |
| `viz_metrics.py` | Visualization of performance metrics |

### Generated Analysis Reports
- `Model-Class_Average_Precision_D.md` — Per-class AP analysis
- `Model-Class_F1-Score_Deltas.md` — F1 score improvements
- `Model-Class_ROC-AUC_Deltas.md` — ROC-AUC analysis
- `Overall_mAP.md` — Summary mAP comparison
- `Per-Class_Metrics.md` — Detailed per-class breakdown

## Technical Insights

### Key Observations

1. **Class frequency correlates with performance**: High-frequency classes (person, car) achieve >0.7 AP while rare classes (fire, smoke) struggle below 0.5 AP. Data augmentation for rare classes partially addresses this.

2. **NMS threshold sensitivity**: The default IoU threshold of 0.5 works well for separated objects but causes missed detections in dense pallet stacks. Per-class NMS thresholds (lower for large objects, higher for small) improved overall mAP by 3%.

3. **Hard negative mining requires careful tuning**: Aggressive mining (low confidence threshold) improves weak classes but degrades strong ones. The optimal approach mines only predictions with confidence 0.3-0.7—high enough to matter, low enough to be uncertain.

### Production Considerations

For deployment at scale:

- **GPU acceleration**: Current OpenCV DNN backend is CPU-only. TensorRT or ONNX Runtime with CUDA would enable 100+ FPS.
- **Batch processing**: Current single-frame inference is inefficient. Batch inference across multiple streams improves throughput.
- **Model versioning**: A/B testing infrastructure for model updates without service disruption.
- **Alert integration**: High-confidence detections of fire/smoke should trigger immediate alerts, not just logging.

### Known Limitations

- YOLOv4-tiny trades accuracy for speed; larger models (YOLOv4, YOLOv5-large) would improve mAP at latency cost
- Single-camera assumption; multi-camera tracking requires additional infrastructure
- No temporal smoothing; frame-to-frame detection flickering in production

## Technologies

- **Object Detection**: YOLOv4-tiny (OpenCV DNN backend)
- **Video Processing**: FFmpeg, OpenCV
- **Containerization**: Docker
- **Metrics**: NumPy, scikit-learn, matplotlib, seaborn

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.
