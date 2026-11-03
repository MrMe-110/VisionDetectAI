
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Load YOLO model
MODEL_NAME = "yolov8n.pt"

model = YOLO(MODEL_NAME)


def detect_image(image, confidence=0.25):
    """
    Detect objects in an image.

    Args:
        image: PIL image
        confidence: Minimum confidence threshold

    Returns:
        annotated_image: Image with bounding boxes
        detections: List of detected objects
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Run YOLO detection
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Generate annotated image
    annotated_array = result.plot()

    # Convert BGR to RGB
    annotated_array = cv2.cvtColor(
        annotated_array,
        cv2.COLOR_BGR2RGB
    )

    annotated_image = Image.fromarray(annotated_array)

    detections = []

    # Extract detection information
    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            object_name = result.names[class_id]

            detections.append({
                "object": object_name,
                "confidence": round(score, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })

    return annotated_image, detections


def get_detection_summary(detections):
    """
    Generate object count and confidence statistics.
    """

    summary = {}

    for detection in detections:

        name = detection["object"]

        summary[name] = summary.get(name, 0) + 1

    total_objects = len(detections)

    if total_objects > 0:

        average_confidence = sum(
            item["confidence"] for item in detections
        ) / total_objects

    else:
        average_confidence = 0

    return {
        "total_objects": total_objects,
        "object_counts": summary,
        "average_confidence": round(
            average_confidence, 4
        )
    }
