
import cv2
import os
import time
import subprocess
import tempfile
from collections import Counter


def process_video(
    input_path,
    output_path,
    confidence=0.25,
    frame_skip=1
):
    from modules.detector import model

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video properties.")

    output_fps = fps / frame_skip

    # Temporary AVI file
    temp_avi = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )
    temp_avi_path = temp_avi.name
    temp_avi.close()

    writer = cv2.VideoWriter(
        temp_avi_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        output_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create temporary video.")

    detections = []
    object_counts = Counter()
    confidence_scores = []

    frame_number = 0
    processed_frames = 0

    start_time = time.time()

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            if (frame_number - 1) % frame_skip != 0:
                continue

            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            result = results[0]

            annotated_frame = result.plot()

            writer.write(annotated_frame)

            timestamp = round(frame_number / fps, 2)

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    object_name = result.names[class_id]

                    detections.append({
                        "frame": frame_number,
                        "timestamp": timestamp,
                        "object": object_name,
                        "confidence": round(score, 4),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    })

                    object_counts[object_name] += 1
                    confidence_scores.append(score)

            processed_frames += 1

    finally:
        cap.release()
        writer.release()

    try:
        # Convert AVI to browser-compatible H.264 MP4
        command = [
            "ffmpeg",
            "-y",
            "-i", temp_avi_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_path
        ]

        conversion = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if conversion.returncode != 0:
            raise RuntimeError(
                "FFmpeg conversion failed: "
                + conversion.stderr[-1500:]
            )

    finally:
        if os.path.exists(temp_avi_path):
            os.remove(temp_avi_path)

    processing_time = time.time() - start_time

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError("Video conversion failed.")

    total_detections = len(detections)

    average_confidence = (
        sum(confidence_scores) / total_detections
        if total_detections else 0
    )

    summary = {
        "total_detections": total_detections,
        "unique_objects": len(object_counts),
        "object_counts": dict(object_counts),
        "average_confidence": round(average_confidence, 4)
    }

    video_info = {
        "original_frames": total_frames,
        "processed_frames": processed_frames,
        "fps": round(fps, 2),
        "duration": round(total_frames / fps, 2),
        "processing_time": round(processing_time, 2),
        "processing_fps": round(
            processed_frames / processing_time, 2
        ) if processing_time > 0 else 0
    }

    return summary, detections, video_info
