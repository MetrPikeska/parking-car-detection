# Parking Car Detection (YOLOv8)

Simple script for running YOLOv8 inference on a video or stream using OpenCV.

## Requirements

- Python 3.9+
- `ultralytics`
- `opencv-python`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install ultralytics opencv-python`
3. Place the model file `yolov8n.pt` in the project root.

## Usage

Edit `video_path` in `car_detect.py` to point to your source:

- Local file: `/path/to/video.mp4`
- Network stream (examples):
  - `rtsp://user:pass@ip:554/stream1`
  - `http://ip:8080/video`
  - `http://ip/stream.m3u8`

Then run:

- `python car_detect.py`

If you are using the provided venv:

- `/home/petr-mikeska/projects/parking-car-detection/venv/bin/python car_detect.py`

## Notes for headless servers

The script is configured to run without any GUI display. It processes frames and performs detection only.

## Output

The script writes an annotated video to `output.mp4` in the project root.
