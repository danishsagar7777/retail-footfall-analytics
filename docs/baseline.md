## Day 2 Optimization Experiment

### RTSP Input

Measured camera FPS:
~20.6 FPS

Camera resolution:
3200x1800

### YOLO Inference

Model:
YOLOv8n

Device:
CPU

Previous YOLO FPS:
6.47 FPS

YOLO FPS with imgsz=640:
~6.00 FPS

Conclusion:
Changing the YOLO inference size to 640 did not provide a significant CPU performance improvement.

OpenVINO optimization:
Next experiment

## Day 2 — OpenVINO Optimization

### OpenVINO Benchmark

Model:
YOLOv8n

Input:
1280x720

Inference size:
640

Device:
CPU

PyTorch YOLO FPS:
~6.00 FPS

OpenVINO YOLO FPS:
16.79 FPS

Speedup:
~2.8x

RTSP camera FPS:
~20.6 FPS

Conclusion:
OpenVINO significantly improves CPU inference performance.
The OpenVINO pipeline reaches approximately 81% of the camera input rate.
## Day 2 — OpenVINO Footfall Pipeline

### OpenVINO Footfall Test

Model:
YOLOv8n

Model format:
OpenVINO

Precision:
FP16

Input resolution:
1280x720

Tracker:
BoT-SORT

Detection class:
Person only

Confidence:
0.40

Counting line:
(136,399) -> (1138,399)

Frames processed:
3536

Pipeline FPS:
14.41 FPS

Footfall result:
Entries: 0
Exits: 2
Occupancy: 0

Conclusion:
The OpenVINO footfall pipeline works correctly on the live RTSP stream.
OpenVINO improves CPU inference performance substantially compared with
the PyTorch baseline.
