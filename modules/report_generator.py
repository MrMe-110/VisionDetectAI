
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


def generate_pdf_report(
    summary,
    detections,
    report_type="Object Detection",
    video_info=None
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0284c7"),
        fontSize=22,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        textColor=colors.grey,
        fontSize=10,
        spaceAfter=20
    )

    story = []

    # TITLE
    story.append(
        Paragraph("VisionDetect AI", title_style)
    )

    story.append(
        Paragraph(
            f"{report_type} Analysis Report",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style
        )
    )

    story.append(Spacer(1, 10))

    # SUMMARY
    story.append(
        Paragraph("Detection Summary", styles["Heading2"])
    )

    total = summary.get(
        "total_objects",
        summary.get("total_detections", 0)
    )

    categories = summary.get(
        "unique_objects",
        len(summary.get("object_counts", {}))
    )

    average = summary.get("average_confidence", 0)

    summary_data = [
        ["Metric", "Value"],
        ["Total Detections", str(total)],
        ["Object Categories", str(categories)],
        ["Average Confidence", f"{average * 100:.2f}%"]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[3 * inch, 2.5 * inch]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 20))

    # VIDEO INFORMATION
    if video_info:

        story.append(
            Paragraph("Video Information", styles["Heading2"])
        )

        video_data = [
            ["Property", "Value"],
            ["Duration", f'{video_info.get("duration", "N/A")} seconds'],
            ["Original FPS", str(video_info.get("fps", "N/A"))],
            ["Processed Frames", str(video_info.get("processed_frames", "N/A"))],
            ["Processing Time", f'{video_info.get("processing_time", "N/A")} seconds']
        ]

        video_table = Table(
            video_data,
            colWidths=[3 * inch, 2.5 * inch]
        )

        video_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ]))

        story.append(video_table)
        story.append(Spacer(1, 20))

    # OBJECT DISTRIBUTION
    story.append(
        Paragraph("Object Distribution", styles["Heading2"])
    )

    object_counts = summary.get("object_counts", {})

    distribution_data = [["Object", "Detection Count"]]

    for object_name, count in object_counts.items():
        distribution_data.append([
            str(object_name),
            str(count)
        ])

    if len(distribution_data) == 1:
        distribution_data.append(["No objects detected", "0"])

    distribution_table = Table(
        distribution_data,
        colWidths=[3 * inch, 2.5 * inch]
    )

    distribution_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(distribution_table)
    story.append(Spacer(1, 20))

    # DETAILED RECORDS
    story.append(
        Paragraph("Detection Records", styles["Heading2"])
    )

    if detections:

        headers = [
            "Object",
            "Confidence",
            "X1",
            "Y1",
            "X2",
            "Y2"
        ]

        if "frame" in detections[0]:
            headers.insert(0, "Frame")

        records = [headers]

        for item in detections[:500]:

            row = []

            if "frame" in item:
                row.append(str(item.get("frame", "")))

            row.extend([
                str(item.get("object", "")),
                f'{item.get("confidence", 0) * 100:.2f}%',
                str(item.get("x1", "")),
                str(item.get("y1", "")),
                str(item.get("x2", "")),
                str(item.get("y2", ""))
            ])

            records.append(row)

        detail_table = Table(
            records,
            repeatRows=1
        )

        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))

        story.append(detail_table)

        if len(detections) > 500:
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "Note: Only the first 500 detection records are included.",
                    styles["Italic"]
                )
            )

    else:

        story.append(
            Paragraph("No detection records available.", styles["Normal"])
        )

    # FOOTER
    def add_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(
            40,
            25,
            "VisionDetect AI | YOLO Object Detection System"
        )
        canvas.drawRightString(
            A4[0] - 40,
            25,
            f"Page {document.page}"
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer
    )

    buffer.seek(0)

    return buffer.getvalue()
