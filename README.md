# Retail Footfall Analytics

A real-time computer vision system for detecting and tracking people from a live RTSP camera stream and measuring footfall using a defined counting line.

The system uses YOLOv8 for person detection, BoT-SORT for object tracking, OpenCV for video processing, and FastAPI for exposing the pipeline status and footfall results through an API.

## Project Overview

Retail stores and similar environments can use camera-based analytics to understand how many people enter and leave a particular area.

This project implements a complete pipeline that:

- Reads video from a live RTSP camera stream
- Detects people using YOLOv8
- Tracks detected people using BoT-SORT
- Assigns persistent tracking IDs to detected people
- Uses a virtual counting line to detect movement across a defined boundary
- Classifies crossings as entries or exits
- Maintains entry, exit, and occupancy counts
- Logs footfall events to a CSV file
- Provides pipeline status and analytics through a FastAPI service
- Handles RTSP connection failures and reconnects automatically
- Processes the latest available frame to avoid building up a large frame backlog

The project is designed as a modular computer vision pipeline so that the detection, tracking, analytics, input, and API components can be developed and tested separately.

---

## System Architecture

```text
                    RTSP Camera Stream
                           |
                           v
                    RTSP Reader
                           |
                           v
                  Latest Frame Selection
                           |
                           v
                  Frame Pre-processing
                           |
                           v
                  YOLOv8 Person Detection
                           |
                           v
                    BoT-SORT Tracking
                           |
                           v
                  Track IDs + Bounding Boxes
                           |
                           v
                     Track Centers
                           |
                           v
                   Counting Line Check
                           |
                    +------+------+
                    |             |
                    v             v
                  Entry          Exit
                    |             |
                    +------+------+
                           |
                           v
                    Footfall Counter
                           |
                    +------+------+
                    |             |
                    v             v
                 CSV Log      FastAPI API
