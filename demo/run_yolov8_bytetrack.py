"""Run a public-safe YOLOv8 + ByteTrack video tracking demo."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 object detection with ByteTrack tracking on a video."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model name or local model path.",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Detection confidence threshold.",
    )
    return parser.parse_args()


def validate_source(source: Path) -> Path:
    """Validate that the input video exists."""
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Input source is not a file: {source}")
    return source


def read_video_properties(source: Path) -> tuple[float, int, int]:
    """Read FPS and frame dimensions from a video file."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {source}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read valid frame dimensions from: {source}")

    if fps <= 0:
        fps = 30.0

    return fps, width, height


def run_tracking(source: Path, output: Path, model_name: str, conf: float) -> None:
    """Run YOLOv8 detection with ByteTrack and save an annotated video."""
    import cv2
    from ultralytics import YOLO

    source = validate_source(source)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    fps, width, height = read_video_properties(source)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create output video: {output}")

    model = YOLO(model_name)
    frame_count = 0

    try:
        results = model.track(
            source=str(source),
            tracker="bytetrack.yaml",
            conf=conf,
            stream=True,
            persist=True,
            verbose=False,
        )

        for result in results:
            annotated_frame = result.plot()
            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                annotated_frame = cv2.resize(annotated_frame, (width, height))
            writer.write(annotated_frame)
            frame_count += 1
    finally:
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were processed from: {source}")

    print(f"Saved annotated tracking video to {output} ({frame_count} frames).")


def main() -> None:
    """Run the command-line demo."""
    args = parse_args()
    run_tracking(args.source, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()
