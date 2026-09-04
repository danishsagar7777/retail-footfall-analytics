import os
import re
import signal
import subprocess
import sys
import threading

from fastapi import FastAPI

from src.api.state import pipeline_state


app = FastAPI(
    title="Retail Footfall Analytics",
    description="Real-time person detection, tracking, and footfall analytics API",
    version="1.0.0",
)


pipeline_process = None
process_lock = threading.Lock()


def read_pipeline_output(process):
    global pipeline_process

    for raw_line in process.stdout:
        line = raw_line.strip()

        if not line:
            continue

        print(f"[PIPELINE] {line}", flush=True)

        if "RTSP stream connected" in line:
            pipeline_state.update(
                camera_connected=True
            )

        if (
            "Loading model:" in line
            or "Loading yolov8" in line
        ):
            pipeline_state.update(
                model_loaded=True
            )

        match = re.search(
            r"Frames:\s*(\d+)\s*\|\s*FPS:\s*([\d.]+)"
            r"\s*\|\s*Entries:\s*(\d+)"
            r"\s*\|\s*Exits:\s*(\d+)"
            r"\s*\|\s*Occupancy:\s*(-?\d+)"
            r"\s*\|\s*Reconnects:\s*(\d+)"
            r"\s*\|\s*Skipped:\s*(\d+)",
            line,
        )

        if match:
            (
                frames,
                fps,
                entries,
                exits,
                occupancy,
                reconnects,
                skipped,
            ) = match.groups()

            pipeline_state.update(
                frames_processed=int(frames),
                fps=float(fps),
                entries=int(entries),
                exits=int(exits),
                occupancy=int(occupancy),
                reconnects=int(reconnects),
                skipped_frames=int(skipped),
            )

        event_match = re.search(
            r"EVENT:\s*track_id=(\d+)\s+type=(entry|exit)",
            line,
        )

        if event_match:
            track_id, event_type = event_match.groups()

            pipeline_state.update(
                last_event={
                    "track_id": int(track_id),
                    "event": event_type,
                }
            )

    return_code = process.wait()

    with process_lock:
        pipeline_state.update(
            running=False
        )
        pipeline_process = None

    print(
        f"Footfall pipeline stopped "
        f"with exit code {return_code}",
        flush=True,
    )


def start_pipeline():
    global pipeline_process

    with process_lock:

        if (
            pipeline_process is not None
            and pipeline_process.poll() is None
        ):
            return False

        pipeline_state.update(
            running=True,
            camera_connected=False,
            model_loaded=False,
            frames_processed=0,
            fps=0.0,
            entries=0,
            exits=0,
            occupancy=0,
            reconnects=0,
            skipped_frames=0,
            last_event=None,
        )

        environment = os.environ.copy()

        # Force Python child process to flush output immediately.
        environment["PYTHONUNBUFFERED"] = "1"

        pipeline_process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.run_footfall",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

        thread = threading.Thread(
            target=read_pipeline_output,
            args=(pipeline_process,),
            daemon=True,
        )

        thread.start()

        print(
            f"Footfall pipeline started "
            f"(PID={pipeline_process.pid})",
            flush=True,
        )

        return True


def stop_pipeline():
    global pipeline_process

    with process_lock:

        if (
            pipeline_process is None
            or pipeline_process.poll() is not None
        ):
            pipeline_state.update(
                running=False
            )
            return False

        process = pipeline_process

        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    return True


@app.on_event("startup")
def startup_event():
    start_pipeline()


@app.on_event("shutdown")
def shutdown_event():
    stop_pipeline()


@app.get("/")
def root():
    return {
        "service": "retail-footfall-analytics",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    state = pipeline_state.snapshot()

    healthy = (
        state["running"]
        and state["camera_connected"]
        and state["model_loaded"]
    )

    return {
        "status": "healthy" if healthy else "unhealthy",
        "camera_connected": state["camera_connected"],
        "model_loaded": state["model_loaded"],
        "pipeline_running": state["running"],
    }


@app.get("/status")
def status():
    return pipeline_state.snapshot()


@app.get("/events")
def events():
    state = pipeline_state.snapshot()

    return {
        "entries": state["entries"],
        "exits": state["exits"],
        "occupancy": state["occupancy"],
        "last_event": state["last_event"],
    }


@app.post("/pipeline/start")
def start():
    started = start_pipeline()

    return {
        "started": started,
        "status": "running"
        if started
        else "already_running",
    }


@app.post("/pipeline/stop")
def stop():
    stopped = stop_pipeline()

    return {
        "stopped": stopped,
        "status": "stopping"
        if stopped
        else "already_stopped",
    }
