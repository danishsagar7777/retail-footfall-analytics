# Retail Footfall Baseline

## Camera

Resolution:
3200x1800

Approximate camera FPS:
~21 FPS

## Processing

Processing resolution:
1280x720

YOLO:
YOLOv8n

Detection class:
Person only

Confidence:
0.40

Tracker:
BoT-SORT

## Counting

Counting line:
(136,399) -> (1138,399)

Dwell:
Not implemented

## Performance

Camera input FPS:
~21 FPS

YOLO-only FPS:
6.47 FPS

Full pipeline FPS:
6.09 FPS

## Footfall Test

Entries:
3

Exits:
0

Occupancy:
3

## Bottleneck

YOLO inference is currently the primary performance bottleneck.

YOLO-only:
6.47 FPS

Full pipeline:
6.09 FPS

Difference:
0.38 FPS
