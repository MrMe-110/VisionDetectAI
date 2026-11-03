# VisionDetect AI — Object Detection System

VisionDetect AI is a Python-based object detection application built with **YOLOv8, OpenCV, and Streamlit**. It detects and identifies objects in images, recorded videos, and live webcam streams, providing real-time analytics and downloadable reports.

## Features

- **Image Detection:** Upload images and detect objects with bounding boxes and confidence scores.
- **Video Detection:** Process recorded videos frame by frame and generate annotated output videos.
- **Live Webcam Detection:** Detect objects through a live camera stream.
- **Webcam Recording:** Record live webcam detection with annotated bounding boxes and download the recorded video.
- **Detection Analytics:** View object counts, confidence scores, and visual detection distributions.
- **PDF Reports:** Generate downloadable PDF detection reports.
- **CSV Export:** Export detection results for further analysis.
- **Session-Based Results:** Keep detection results within the current user session without storing shared detection history.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 (Ultralytics) | Object detection |
| OpenCV | Image and video processing |
| Streamlit | Interactive web interface |
| Pandas & NumPy | Data processing |
| Matplotlib | Data visualization |
| ReportLab | PDF report generation |
| Streamlit-WebRTC & PyAV | Live webcam streaming |
| FFmpeg | Video conversion and encoding |

## How It Works

1. The user selects an input mode: image, video, or webcam.
2. YOLOv8 Nano processes the input and identifies objects.
3. Detected objects are displayed with bounding boxes and confidence scores.
4. Detection results are summarized through analytics and visualizations.
5. Users can download annotated images, processed videos, CSV results, and PDF reports where supported.
6. Live webcam users can record and download annotated video footage.

Detection results are session-based. The application does not use a shared SQLite database or persistent detection history.

## Project Structure

```text
VisionDetectAI/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
│
├── models/
│
├── modules/
│   ├── __init__.py
│   ├── detector.py
│   ├── video_processor.py
│   ├── webcam_processor.py
│   └── report_generator.py
│
└── output/
    ├── detections/
    │   └── .gitkeep
    └── reports/
        └── .gitkeep
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd VisionDetectAI
```

### 2. Install dependencies

```bash
py -m pip install -r requirements.txt
```

Install FFmpeg separately if it is not already available on your system.

### 3. Run the application

```bash
py -m streamlit run app.py
```

The application will open in your browser.

## Model Information

This project uses the pretrained **YOLOv8 Nano (`yolov8n.pt`)** model from Ultralytics. It supports object detection across the model's pretrained COCO classes.

The model weights are downloaded automatically when the application first loads the model.

## Future Improvements

- Custom dataset training and fine-tuning.
- Object tracking across video frames.
- Real-time detection performance improvements.
- Cloud-based deployment and scalability.
- Advanced analytics and reporting.

## Project Status

Developed as an AI and Computer Vision portfolio project to explore object detection, video processing, deep learning model integration, and interactive application development.

## Author

**Your Name**
