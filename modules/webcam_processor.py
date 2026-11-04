
import cv2
import av
import os
import tempfile
import threading
from collections import Counter
from streamlit_webrtc import VideoProcessorBase

# Load the YOLO model once when the processor is created.
from modules.detector import model


class WebcamProcessor(VideoProcessorBase):

    INFERENCE_INTERVAL = 2
    MAX_FRAME_SIZE = 640
    YOLO_IMAGE_SIZE = 480
    CONFIDENCE = 0.35

    def __init__(self):
        self.lock = threading.Lock()

        self.object_counts = Counter()
        self.total_detections = 0
        self.frame_count = 0

        # Memory-efficient confidence tracking
        self.confidence_sum = 0.0
        self.confidence_count = 0

        self.last_annotated = None

        self.recording_requested = False
        self.video_writer = None
        self.recording_path = None
        self.recording_error = None

    def start_recording(self):

        with self.lock:
            if self.video_writer is not None:
                return False

            self.recording_requested = True
            self.recording_error = None

        return True

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        original_height, original_width = img.shape[:2]

        with self.lock:
            self.frame_count += 1
            current_frame = self.frame_count

        # Run YOLO on every second frame.
        should_infer = (
            self.last_annotated is None
            or current_frame % self.INFERENCE_INTERVAL == 1
        )

        if should_infer:

            # Resize large camera frames before inference.
            scale = min(
                1.0,
                self.MAX_FRAME_SIZE / max(original_width, original_height)
            )

            if scale < 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                inference_img = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )
            else:
                inference_img = img

            results = model.predict(
                source=inference_img,
                conf=self.CONFIDENCE,
                imgsz=self.YOLO_IMAGE_SIZE,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            # Restore the original camera dimensions.
            if annotated.shape[:2] != (original_height, original_width):
                annotated = cv2.resize(
                    annotated,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR
                )

            frame_objects = []

            if result.boxes is not None:
                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    object_name = result.names[class_id]

                    frame_objects.append((object_name, score))

            self.last_annotated = annotated

            with self.lock:
                self.total_detections += len(frame_objects)

                for name, score in frame_objects:
                    self.object_counts[name] += 1
                    self.confidence_sum += score
                    self.confidence_count += 1

        else:
            # Reuse the latest annotated image.
            annotated = self.last_annotated

        # Handle video recording.
        with self.lock:

            if self.recording_requested:

                if self.video_writer is None:

                    height, width = annotated.shape[:2]

                    fd, self.recording_path = tempfile.mkstemp(
                        suffix=".avi"
                    )
                    os.close(fd)

                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

                    self.video_writer = cv2.VideoWriter(
                        self.recording_path,
                        fourcc,
                        20.0,
                        (width, height)
                    )

                    if not self.video_writer.isOpened():

                        self.video_writer.release()
                        self.video_writer = None
                        self.recording_requested = False
                        self.recording_error = (
                            "Could not initialize video recording."
                        )

                if self.video_writer is not None:
                    self.video_writer.write(annotated)

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

    def stop_recording(self):

        with self.lock:

            self.recording_requested = False

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            path = self.recording_path
            self.recording_path = None

            error = self.recording_error

        if not path or not os.path.exists(path):
            return None, error

        try:
            with open(path, "rb") as video_file:
                video_bytes = video_file.read()

            return video_bytes, None

        except Exception as error:
            return None, str(error)

        finally:
            if os.path.exists(path):
                os.remove(path)

    def get_statistics(self):

        with self.lock:

            average = (
                self.confidence_sum / self.confidence_count
                if self.confidence_count else 0
            )

            return {
                "frames": self.frame_count,
                "total_detections": self.total_detections,
                "object_counts": dict(self.object_counts),
                "average_confidence": round(average, 4),
                "recording": self.recording_requested,
                "recording_error": self.recording_error
            }
