# Retail Footfall Baseline

## Hardware

- Device: Lenovo IdeaPad Slim 5
- CPU inference
- GPU acceleration: Not available

## Model

- Detector: YOLO
- Backend: OpenVINO
- Input size: 640x640
- Detection class: person

## Pipeline

RTSP Camera
→ Frame Processing
→ YOLO Detection
→ Tracking
→ Line Crossing
→ Footfall Counter
→ CSV Event Logger

## Baseline Performance

- Approximate FPS: 16
- RTSP reconnects: 0
- OpenVINO inference: Working
- Footfall events: Working

## Known Limitations

- Accuracy has not yet been formally evaluated.
- Coordinate consistency must be verified.
- Long-duration reliability testing is pending.
- RTSP recovery testing is pending.
