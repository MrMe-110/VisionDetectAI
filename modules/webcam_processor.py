
import cv2
import av
import os
import tempfile
import threading
from collections import Counter
from streamlit_webrtc import VideoProcessorBase


class WebcamProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()

        self.object_counts = Counter()
        self.total_detections = 0
        self.frame_count = 0
        self.confidence_scores = []

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

        from modules.detector import model

        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=img,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        frame_objects = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                object_name = result.names[class_id]

                frame_objects.append((object_name, score))

        with self.lock:

            self.frame_count += 1
            self.total_detections += len(frame_objects)

            for name, score in frame_objects:
                self.object_counts[name] += 1
                self.confidence_scores.append(score)

            # RECORD ANNOTATED FRAMES

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
                        self.recording_error = "Could not initialize video recording."

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

            counts = dict(self.object_counts)
            total = self.total_detections

            average = (
                sum(self.confidence_scores) / len(self.confidence_scores)
                if self.confidence_scores else 0
            )

            return {
                "frames": self.frame_count,
                "total_detections": total,
                "object_counts": counts,
                "average_confidence": round(average, 4),
                "recording": self.recording_requested,
                "recording_error": self.recording_error
            }
