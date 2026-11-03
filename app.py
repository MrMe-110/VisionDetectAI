
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import tempfile
import os

from PIL import Image
from io import BytesIO

from modules.detector import detect_image, get_detection_summary
from modules.video_processor import process_video
from modules.webcam_processor import WebcamProcessor
from modules.report_generator import generate_pdf_report

from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="VisionDetect AI",
    page_icon="👁️",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.title("👁️ VisionDetect AI")

st.subheader(
    "Intelligent Object Detection & Analysis System"
)

st.write(
    "Detect, analyze, and generate insights from "
    "images, recorded videos, and live webcam using YOLO."
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Detection Settings")

confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.10,
    0.95,
    0.25,
    0.05
)

st.sidebar.info(
    "Higher confidence filters out weaker detections."
)

st.sidebar.markdown("---")

st.sidebar.write("**Detection Model:** YOLOv8 Nano")

st.sidebar.write("**Supported Inputs:**")
st.sidebar.write("- Images")
st.sidebar.write("- Recorded Videos")
st.sidebar.write("- Live Webcam")

st.sidebar.markdown("---")

st.sidebar.caption("VisionDetect AI | Computer Vision Project")


# ==================================================
# HELPER FUNCTION: DISPLAY ANALYTICS
# ==================================================

def display_analytics(counts, chart_title):

    if not counts:
        st.info("No objects detected.")
        return

    chart_col, table_col = st.columns(2)

    with chart_col:

        fig, ax = plt.subplots(figsize=(7, 4))

        objects = list(counts.keys())
        values = list(counts.values())

        ax.bar(objects, values)

        ax.set_title(chart_title)
        ax.set_ylabel("Detection Count")
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with table_col:

        stats_df = pd.DataFrame(
            list(counts.items()),
            columns=["Object", "Count"]
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# MAIN TABS
# ==================================================

image_tab, video_tab, webcam_tab = st.tabs([
    "📷 Image Detection",
    "🎬 Video Detection",
    "📹 Live Webcam"
])


# ==================================================
# IMAGE DETECTION
# ==================================================

with image_tab:

    st.header("Image Object Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload"
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🚀 Detect Objects",
            type="primary",
            key="image_detect"
        ):

            with st.spinner("Analyzing image..."):

                start_time = time.time()

                annotated_image, detections = detect_image(
                    image,
                    confidence
                )

                processing_time = time.time() - start_time

                summary = get_detection_summary(detections)

                st.session_state["image_result"] = {
                    "annotated": annotated_image,
                    "detections": detections,
                    "summary": summary,
                    "time": processing_time
                }

    # IMAGE RESULTS

    if "image_result" in st.session_state:

        result = st.session_state["image_result"]

        summary = result["summary"]
        detections = result["detections"]

        st.divider()

        st.header("📊 Image Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Objects",
            summary["total_objects"]
        )

        col2.metric(
            "Object Categories",
            len(summary["object_counts"])
        )

        col3.metric(
            "Average Confidence",
            f'{summary["average_confidence"] * 100:.2f}%'
        )

        st.subheader("Detected Image")

        st.image(
            result["annotated"],
            use_container_width=True
        )

        st.subheader("Object Distribution")

        display_analytics(
            summary["object_counts"],
            "Image Object Distribution"
        )

        st.subheader("Detailed Detection Records")

        if detections:

            df = pd.DataFrame(detections)

            display_df = df.copy()

            display_df["confidence"] = (
                display_df["confidence"] * 100
            ).round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Image CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="image_detection_results.csv",
                mime="text/csv"
            )

        else:

            st.info("No objects detected.")

        # ANNOTATED IMAGE DOWNLOAD

        image_buffer = BytesIO()

        result["annotated"].save(
            image_buffer,
            format="JPEG"
        )

        st.download_button(
            "⬇️ Download Annotated Image",
            data=image_buffer.getvalue(),
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )

        # IMAGE PDF REPORT

        pdf_report = generate_pdf_report(
            summary=summary,
            detections=detections,
            report_type="Image Detection"
        )

        st.download_button(
            "📄 Download Image PDF Report",
            data=pdf_report,
            file_name="visiondetect_image_report.pdf",
            mime="application/pdf",
            key="image_pdf_report"
        )

        st.success(
            f'Analysis completed in {result["time"]:.2f} seconds.'
        )

    else:

        st.info("Upload an image and click Detect Objects.")


# ==================================================
# VIDEO DETECTION
# ==================================================

with video_tab:

    st.header("Recorded Video Object Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    frame_skip = st.selectbox(
        "Frame Processing Interval",
        [1, 2, 3, 5],
        index=0
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button(
            "🚀 Process Video",
            type="primary",
            key="video_process"
        ):

            input_path = None
            output_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_input:

                    temp_input.write(
                        uploaded_video.getbuffer()
                    )

                    input_path = temp_input.name

                temp_output = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_path = temp_output.name
                temp_output.close()

                with st.spinner(
                    "YOLO is processing your video..."
                ):

                    summary, detections, video_info = process_video(
                        input_path,
                        output_path,
                        confidence,
                        frame_skip
                    )

                    with open(output_path, "rb") as file:

                        processed_video = file.read()

                    st.session_state["video_result"] = {
                        "summary": summary,
                        "detections": detections,
                        "info": video_info,
                        "video": processed_video
                    }

                st.success("Video processing completed!")

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )

            finally:

                for path in [input_path, output_path]:

                    if path and os.path.exists(path):
                        os.remove(path)

    # VIDEO RESULTS

    if "video_result" in st.session_state:

        result = st.session_state["video_result"]

        summary = result["summary"]
        detections = result["detections"]
        info = result["info"]

        st.divider()

        st.header("📊 Video Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Detections",
            summary["total_detections"]
        )

        col2.metric(
            "Object Categories",
            summary["unique_objects"]
        )

        col3.metric(
            "Average Confidence",
            f'{summary["average_confidence"] * 100:.2f}%'
        )

        st.subheader("Processed Video")

        st.video(result["video"])

        st.download_button(
            "⬇️ Download Processed Video",
            data=result["video"],
            file_name="visiondetect_processed.mp4",
            mime="video/mp4"
        )

        st.subheader("Video Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Duration",
            f'{info["duration"]} sec'
        )

        col2.metric(
            "Original FPS",
            info["fps"]
        )

        col3.metric(
            "Processed Frames",
            info["processed_frames"]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Processing Time",
            f'{info["processing_time"]} sec'
        )

        col2.metric(
            "Processing Speed",
            f'{info["processing_fps"]} FPS'
        )

        st.subheader("Object Distribution")

        display_analytics(
            summary["object_counts"],
            "Video Object Distribution"
        )

        st.subheader("Frame-Level Detection Records")

        if detections:

            video_df = pd.DataFrame(detections)

            display_df = video_df.copy()

            display_df["confidence"] = (
                display_df["confidence"] * 100
            ).round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Video CSV",
                data=video_df.to_csv(index=False).encode("utf-8"),
                file_name="video_detection_results.csv",
                mime="text/csv"
            )

        else:

            st.info("No objects detected in this video.")

        # VIDEO PDF REPORT

        pdf_report = generate_pdf_report(
            summary=summary,
            detections=detections,
            report_type="Video Detection",
            video_info=info
        )

        st.download_button(
            "📄 Download Video PDF Report",
            data=pdf_report,
            file_name="visiondetect_video_report.pdf",
            mime="application/pdf",
            key="video_pdf_report"
        )

    else:

        st.info("Upload a video and click Process Video.")



# ==================================================
# LIVE WEBCAM DETECTION
# ==================================================

with webcam_tab:

    st.header("📹 Live Webcam Detection")

    st.write(
        "Detect objects continuously using your computer's webcam."
    )

    st.info(
        "Click START and allow camera access when your browser asks."
    )

    webrtc_ctx = webrtc_streamer(
        key="visiondetect-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if webrtc_ctx.state.playing:
        st.success("Webcam detection is running.")
    else:
        st.info("Click START to activate your webcam.")

    processor = webrtc_ctx.video_processor

    if processor:

        st.divider()

        # RECORDING CONTROLS

        st.subheader("🎥 Webcam Recording")

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔴 Start Recording",
                disabled=not webrtc_ctx.state.playing,
                key="start_webcam_recording"
            ):

                processor.start_recording()
                st.session_state.pop("webcam_recording", None)
                st.rerun()

        with col2:

            if st.button(
                "⏹️ Stop Recording",
                disabled=not processor.get_statistics()["recording"],
                key="stop_webcam_recording"
            ):

                video_bytes, error = processor.stop_recording()

                if video_bytes:
                    st.session_state["webcam_recording"] = video_bytes
                    st.success("Recording completed!")

                elif error:
                    st.error(error)

                else:
                    st.warning("No recording frames were captured.")

                st.rerun()

        stats = processor.get_statistics()

        if stats["recording"]:
            st.warning("Recording is in progress...")

        if stats["recording_error"]:
            st.error(stats["recording_error"])

        # DOWNLOAD RECORDED VIDEO

        if "webcam_recording" in st.session_state:

            st.subheader("Recorded Webcam Video")

            st.download_button(
                "⬇️ Download Recorded Video",
                data=st.session_state["webcam_recording"],
                file_name="visiondetect_webcam_recording.avi",
                mime="video/x-msvideo",
                key="download_webcam_video"
            )

        st.divider()

        # LIVE STATISTICS

        st.subheader("Live Detection Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Frames Processed", stats["frames"])

        col2.metric(
            "Total Detections",
            stats["total_detections"]
        )

        col3.metric(
            "Average Confidence",
            f'{stats["average_confidence"] * 100:.2f}%'
        )

        if st.button(
            "🔄 Refresh Webcam Statistics",
            key="refresh_webcam"
        ):
            st.rerun()

        st.subheader("Detected Objects")

        if stats["object_counts"]:

            display_analytics(
                stats["object_counts"],
                "Webcam Detection Distribution"
            )

            # SAVE STATISTICS AS CSV

            webcam_df = pd.DataFrame(
                list(stats["object_counts"].items()),
                columns=["Object", "Detection Count"]
            )

            st.download_button(
                "⬇️ Save Webcam Detection Results (CSV)",
                data=webcam_df.to_csv(index=False).encode("utf-8"),
                file_name="webcam_detection_results.csv",
                mime="text/csv",
                key="download_webcam_csv"
            )

        else:
            st.info("No objects detected yet.")

        st.caption(
            "Counts are cumulative detections across processed frames, "
            "not unique physical objects."
        )



# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "VisionDetect AI | Powered by YOLO, OpenCV and Streamlit"
)
