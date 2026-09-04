# Footfall Evaluation

## Environment

- Camera: RTSP
- Detector: YOLO
- Backend: OpenVINO
- Input: 640x640
- Hardware: CPU
- Approximate FPS: 15.49

## Functional Tests

| Test | Actual | Detected | Result |
|---|---:|---:|---|
| Single entry | 1 | 1 | PASS |
| Single exit | 1 | 1 | PASS |
| Multiple entry | 3 | - | - |
| False crossing | 0 | - | - |
| Repeated crossing | - | - | - |
| Occlusion | - | - | - |

## Reliability

- 30-minute runtime: pending
- RTSP reconnects: pending
- Crashes: pending
- Clean shutdown: pending

## Known Limitations

- Accuracy depends on camera angle and lighting.
- Tracking failures can affect footfall counts.
- Skipped frames occur because the reader prioritizes the newest frame.
