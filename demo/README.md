# YOLOv8 + ByteTrack Video Tracking Demo

This directory contains a public-safe engineering demonstration of the computer-vision front end used in the broader video AI pipeline.

This demo illustrates generic detection and tracking functionality and does not implement the research-specific video selection method used in the thesis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Users provide their own input video. No sample media, private footage, or thesis dataset is included.

## Run

```bash
python demo/run_yolov8_bytetrack.py \
    --source path/to/video.mp4 \
    --output outputs/tracked.mp4
```

Optional arguments:

```bash
python demo/run_yolov8_bytetrack.py \
    --source path/to/video.mp4 \
    --output outputs/tracked.mp4 \
    --model yolov8n.pt \
    --conf 0.25
```

## What It Demonstrates

- Video decoding
- YOLOv8 inference
- ByteTrack tracking
- Track ID visualization
- Class-label visualization
- Annotated video export
