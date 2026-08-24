"""Generate the demo corpus's visual half.

`data/assets/` shipped empty, so CLIP, the visual store and the visual
specialist were never exercised against real data and `modality_match` was
capped at 0.200 no matter how good retrieval got.

These are generated demo assets, not real business records. They are derived
from the same figures as data/orders.csv so the visual and text halves agree.

    python scripts/make_demo_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "data" / "assets"

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
ON_TIME = [94, 88, 71, 63]
SUPPLIERS = ["GlobalTech Supply", "PrimeSource Industrial",
             "Pacific Rim Electronics", "Apex Materials"]
DELAYS = [3, 5, 9, 27]


def quarterly_chart(path: Path) -> None:
    """Bar chart behind 'What does the chart in the quarterly report show?'."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(QUARTERS, ON_TIME, color=["#3d7ea6"] * 3 + ["#c0392b"])
    ax.set_title("Quarterly Report — On-Time Delivery Rate by Quarter")
    ax.set_ylabel("On-time delivery (%)")
    ax.set_ylim(0, 100)
    for bar, value in zip(bars, ON_TIME):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value}%",
                ha="center")
    ax.text(0.5, -0.18, "On-time delivery declined from 94% in Q1 to 63% in Q4.",
            transform=ax.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def supplier_delay_chart(path: Path) -> None:
    """Which supplier has the most problems, as a picture."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(SUPPLIERS, DELAYS, color=["#5d8aa8"] * 3 + ["#c0392b"])
    ax.set_title("Delayed Dispatches by Supplier (2023)")
    ax.set_xlabel("Delayed dispatches")
    for i, value in enumerate(DELAYS):
        ax.text(value + 0.4, i, str(value), va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def architecture_diagram(path: Path) -> None:
    """Boxes-and-arrows image behind 'Describe the architecture diagram'."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_title("Supply Chain System Architecture")
    boxes = [
        (0.02, "Suppliers"), (0.22, "Warehouses"), (0.42, "Dispatch"),
        (0.62, "Carriers"), (0.82, "Customers"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 0.45), 0.16, 0.22, facecolor="#dce6f1",
                                   edgecolor="#2c3e50"))
        ax.text(x + 0.08, 0.56, label, ha="center", va="center", fontsize=10)
    for x, _ in boxes[:-1]:
        ax.annotate("", xy=(x + 0.20, 0.56), xytext=(x + 0.18, 0.56),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax.text(0.5, 0.22, "Orders flow left to right; delays propagate downstream.",
            ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def scanned_notice(path: Path) -> None:
    """A text-bearing image for the OCR case."""
    img = Image.new("RGB", (900, 560), "#f6f3ec")
    draw = ImageDraw.Draw(img)
    lines = [
        "SUPPLIER PERFORMANCE NOTICE",
        "",
        "To:      Procurement Department",
        "From:    Quality Assurance",
        "Subject: Apex Materials — corrective action required",
        "",
        "Apex Materials recorded 27 delayed dispatches during 2023,",
        "the highest of any supplier. Root causes include warehouse",
        "inventory discrepancies and repeated recount delays.",
        "",
        "Action: escalate to Level 2 and review contract terms.",
    ]
    y = 40
    for line in lines:
        draw.text((50, y), line, fill="#1a1a1a")
        y += 34
    draw.rectangle([20, 20, 880, 540], outline="#888", width=2)
    img.save(path)


def supplier_workbook(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Supplier Metrics"
    ws.append(["Supplier", "Orders", "Delayed", "On-time %", "Region"])
    rows = [
        ("GlobalTech Supply", 62, 3, 95.2, "North"),
        ("PrimeSource Industrial", 48, 5, 89.6, "West"),
        ("Pacific Rim Electronics", 39, 9, 76.9, "East"),
        ("Apex Materials", 26, 27, 51.9, "East"),
    ]
    for row in rows:
        ws.append(row)
    wb.save(path)


def quarterly_deck(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Quarterly Supply Chain Review"
    slide.placeholders[1].text = "On-time delivery, supplier performance, actions"

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Key Takeaways"
    body = slide.placeholders[1].text_frame
    body.text = "On-time delivery fell from 94% in Q1 to 63% in Q4"
    for line in [
        "Apex Materials accounts for 27 of 44 delayed dispatches",
        "East region fulfilment trails West by 13 points",
        "Machine downtime rose after bearing shipments were delayed",
    ]:
        body.add_paragraph().text = line

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Recommended Actions"
    tb = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(3))
    tb.text_frame.text = ("Escalate Apex Materials to Level 2 review; "
                          "dual-source spindle bearings; add weekly dispatch audits.")
    prs.save(path)


BUILDERS = [
    ("quarterly_report_chart.png", quarterly_chart),
    ("supplier_delays_chart.png", supplier_delay_chart),
    ("architecture_diagram.png", architecture_diagram),
    ("scanned_supplier_notice.png", scanned_notice),
    ("supplier_metrics.xlsx", supplier_workbook),
    ("quarterly_review.pptx", quarterly_deck),
]


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, build in BUILDERS:
        target = ASSETS / name
        build(target)
        print(f"  {name:32s} {target.stat().st_size / 1024:7.1f} KB")
    print(f"{len(BUILDERS)} assets written to {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
