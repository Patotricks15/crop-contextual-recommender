"""Generate the two architecture diagrams as editable draw.io files.

Run with a dedicated venv so drawpyo does not leak into the project deps:

    python -m venv .diagram-venv && .diagram-venv/bin/pip install drawpyo
    .diagram-venv/bin/python scripts/generate_diagrams.py

Then rasterize with the draw.io exporter:

    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f svg -o . --output-mode relative --remove-page-suffix .
    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f png -o . --output-mode relative --remove-page-suffix .
    docker run --rm -v "$PWD/docs":/data alpine chown -R "$(id -u):$(id -g)" /data
    python scripts/generate_diagrams.py --patch-svg-background
"""

from __future__ import annotations

import re
from pathlib import Path

import drawpyo
from drawpyo.diagram import Edge

DOCS = Path(__file__).resolve().parents[1] / "docs"

AWS_SHAPE = (
    "sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#{fill};"
    "strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;"
    "align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;"
    "shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{icon};"
)
BOX = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#{fill};strokeColor=#{stroke};"
    "fontSize=11;fontColor=#232F3E;align=center;verticalAlign=middle;"
)

ORANGE, BLUE, GREEN, PURPLE = "D45B07", "4D72F3", "277116", "8C4FFF"


def aws_node(page, title, icon, fill, x, y):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(AWS_SHAPE.format(fill=fill, icon=icon))
    node.position = (x, y)
    node.width, node.height = 78, 78
    return node


def box(page, title, x, y, width=190, height=58, fill="FFFFFF", stroke="9AA5B1"):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(BOX.format(fill=fill, stroke=stroke))
    node.position = (x, y)
    node.width, node.height = width, height
    return node


def link(page, source, target, label=""):
    return Edge(
        page=page,
        source=source,
        target=target,
        label=label,
        fontSize=10,
        fontColor="#4A5568",
        strokeColor="#4A5568",
        endArrow="blockThin",
        rounded=1,
    )


def build_aws_architecture() -> None:
    """Offline replay-training on the left, stateless online scoring on the right."""
    file = drawpyo.File(file_name="aws-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="AWS architecture")

    box(
        page, "Offline — scheduled replay training", 30, 20, 480, 36, fill="FFF4E6", stroke="D45B07"
    )
    box(
        page, "Online — stateless scoring (p99 < 50 ms)",
        590, 20, 480, 36, fill="EEF2FF", stroke="4D72F3",
    )

    raw_source = box(page, "Soil/climate rounds\nN,P,K,temp,humidity,pH,rainfall", 30, 100, 220, 58)
    raw = aws_node(page, "S3\nraw", "bucket", GREEN, 320, 90)
    train = aws_node(page, "Lambda\nLinUCB replay fit", "lambda_function", ORANGE, 320, 260)
    schedule = aws_node(page, "EventBridge\nscheduled trigger", "event_time_based", PURPLE, 90, 260)
    artifacts = aws_node(page, "S3\npolicy.json (A, b, scaler)", "bucket", GREEN, 320, 430)

    client = aws_node(page, "Client\n(soil/climate reading)", "client", BLUE, 620, 90)
    api = aws_node(page, "API Gateway", "api_gateway", PURPLE, 840, 90)
    lam = aws_node(page, "Lambda\nconstrained scoring", "lambda_function", ORANGE, 840, 260)
    logs = aws_node(page, "CloudWatch\nlogs & metrics", "cloudwatch_2", PURPLE, 1030, 430)

    link(page, raw_source, raw, "ingest")
    link(page, schedule, train, "cron")
    link(page, raw, train, "read")
    link(page, train, artifacts, "publish policy")
    link(page, client, api, "GET /recommend")
    link(page, api, lam, "invoke")
    link(page, artifacts, lam, "GetObject (cold start)")
    link(page, lam, logs, "emit")

    file.write()
    print(f"wrote {DOCS / 'aws-architecture.drawio'}")


def build_recommendation_architecture() -> None:
    """The modelling pipeline, from a raw context to a constrained LinUCB choice."""
    file = drawpyo.File(file_name="recommendation-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="Recommendation architecture")

    spine_x = 420
    w, gap = 280, 118

    raw = box(
        page, "Raw soil/climate round\nN, P, K, temp, humidity, pH, rainfall",
        spine_x, 40, w, 72, fill="F7FAFC",
    )
    labelled = box(
        page, "Simulated bandit reward\nreward = 1 iff arm == logged crop",
        spine_x, 40 + gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    split = box(
        page, "Row-level random holdout\ntrain = replay stream",
        spine_x, 40 + 2 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    scaler = box(
        page, "Context z-score + arm feasible range\nfit on training rounds only",
        spine_x, 40 + 3 * gap, w, 72, fill="E6FFFA", stroke="277116",
    )
    constraint = box(
        page, "Agronomic feasibility filter\nexclude arms outside observed range",
        spine_x, 55 + 4 * gap, w, 58, fill="EEF2FF", stroke="4D72F3",
    )
    linucb = box(
        page,
        "LinUCB\ntheta_a = A_a^-1 b_a\nUCB = x.theta + alpha*sqrt(x.A^-1.x)",
        spine_x, 55 + 5 * gap, w, 86, fill="EEF2FF", stroke="4D72F3",
    )
    evaluation = box(
        page,
        "Evaluation\nonline learning curve + frozen match rate/regret",
        spine_x, 70 + 6 * gap, w, 72, fill="FFF5F5", stroke="C53030",
    )

    baselines = box(
        page, "Baselines\nRandom · Epsilon-greedy popularity",
        spine_x - 350, 55 + 5 * gap + 14, 260, 58,
    )
    serving = box(
        page, "Policy serialization\nA, b, scaler, ranges -> JSON",
        spine_x + 350, 70 + 6 * gap - 14, 260, 58, fill="EEF2FF", stroke="4D72F3",
    )

    link(page, raw, labelled)
    link(page, labelled, split)
    link(page, split, scaler, "train only")
    link(page, scaler, constraint)
    link(page, constraint, linucb, "feasible arms")
    link(page, linucb, evaluation, "chosen arm + reward")
    link(page, baselines, linucb)
    link(page, linucb, serving, "fitted policy")

    file.write()
    print(f"wrote {DOCS / 'recommendation-architecture.drawio'}")


def patch_svg_white_background(svg_path: Path) -> None:
    """Insert a literal white background rect into an exported SVG."""
    text = svg_path.read_text()
    width_match = re.search(r'width="(\d+)px"', text)
    height_match = re.search(r'height="(\d+)px"', text)
    if not (width_match and height_match):
        raise ValueError(f"{svg_path} has no pixel width/height to size the background rect.")
    width, height = width_match.group(1), height_match.group(1)
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'

    if rect in text:
        return

    anchor = "<defs/><g>"
    if anchor not in text:
        raise ValueError(f"{svg_path} does not match the expected <defs/><g> export shape.")

    patched = text.replace(anchor, anchor + rect, 1)
    svg_path.write_text(patched)
    print(f"patched white background into {svg_path}")


if __name__ == "__main__":
    import sys

    DOCS.mkdir(exist_ok=True)
    if "--patch-svg-background" in sys.argv:
        for svg in sorted(DOCS.glob("*.svg")):
            patch_svg_white_background(svg)
    else:
        build_aws_architecture()
        build_recommendation_architecture()
