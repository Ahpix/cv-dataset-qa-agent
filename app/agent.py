# app/agent.py
"""
Computer Vision Dataset Advisor + Label QA Agent
Google ADK v2.3.0
Dataset-path based implementation.

This version performs deterministic dataset analysis locally and
uses Gemini only for generating the final natural-language report.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

import cv2
import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google.adk.workflow import Workflow, START
from google.adk.agents.llm_agent import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)

LIVE_MODE = bool(os.getenv("GEMINI_API_KEY"))

# ------------------------------------------------------------------
# Gemini schema helper
# ------------------------------------------------------------------

def remove_additional_properties(schema: dict) -> None:
    schema.pop("additionalProperties", None)
    schema.pop("additional_properties", None)

    if "properties" in schema:
        for value in schema["properties"].values():
            if isinstance(value, dict):
                remove_additional_properties(value)

    if "$defs" in schema:
        for value in schema["$defs"].values():
            if isinstance(value, dict):
                remove_additional_properties(value)

# ------------------------------------------------------------------
# Output Schema
# ------------------------------------------------------------------

class ClassDistributionItem(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": remove_additional_properties,
    }

    class_name: str
    count: int


class LabelIssueItem(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": remove_additional_properties,
    }

    issue_type: str
    count: int


class CVReportSchema(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": remove_additional_properties,
    }

    dataset_name: str
    total_images: int
    average_resolution: str

    class_distribution: List[ClassDistributionItem]

    vision_quality_score: float

    label_issues: List[LabelIssueItem]

    recommendations: List[str]

# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".webp",
}

BLUR_THRESHOLD = 100.0


def find_key(data: Any, target: str):

    if isinstance(data, dict):

        if target in data:
            return data[target]

        for value in data.values():
            result = find_key(value, target)
            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:
            result = find_key(item, target)
            if result is not None:
                return result

    return None





# ------------------------------------------------------------------
# Input Parsing
# ------------------------------------------------------------------

def parse_dataset_metadata(ctx, node_input=None) -> Dict[str, Any]:
    """
    Expected input:

    {
        "dataset_path": "...",
        "dataset_metadata": {
            "dataset_name": "..."
        }
    }
    """

    raw_text = ""

    if hasattr(node_input, "parts"):
        for part in node_input.parts:
            if hasattr(part, "text"):
                raw_text += part.text

    elif isinstance(node_input, str):
        raw_text = node_input

    else:
        raw_text = str(node_input)

    import re

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if match is None:
        raise ValueError("No JSON payload found.")

    payload = json.loads(match.group())

    dataset_path = find_key(payload, "dataset_path")

    metadata = find_key(payload, "dataset_metadata")

    if metadata is None:
        metadata = {}

    if dataset_path is None:
        raise ValueError("dataset_path is required.")

    dataset_path = str(Path(dataset_path).expanduser())

    metadata["dataset_path"] = dataset_path

    if "dataset_name" not in metadata:
        metadata["dataset_name"] = Path(dataset_path).name

    ctx.state["metadata"] = metadata

    return {"metadata": metadata}


# ------------------------------------------------------------------
# Dataset Scanner
# ------------------------------------------------------------------

def scan_dataset(dataset_path: str) -> Dict[str, Any]:

    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(dataset_path)

    class_counter = Counter()

    total_images = 0

    width_sum = 0
    height_sum = 0

    brightness_values = []

    blurry_images = 0
    corrupted_images = 0

    hash_counter = Counter()

    duplicate_images = 0

    invalid_files = 0

    empty_class_folders = 0

    for class_dir in sorted(root.iterdir()):

        if not class_dir.is_dir():
            continue

        class_count = 0

        for file in class_dir.rglob("*"):

            if not file.is_file():
                continue

            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                invalid_files += 1
                continue

            try:

                image = Image.open(file).convert("RGB")

            except (UnidentifiedImageError, OSError):

                corrupted_images += 1
                continue

            width, height = image.size

            width_sum += width
            height_sum += height

            total_images += 1
            class_count += 1

            gray = np.array(image.convert("L"))

            brightness_values.append(float(gray.mean()))

            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian < BLUR_THRESHOLD:
                blurry_images += 1

            image_hash = str(imagehash.phash(image))

            hash_counter[image_hash] += 1

        if class_count == 0:

            empty_class_folders += 1

        else:

            class_counter[class_dir.name] = class_count

    duplicate_images = sum(
        count - 1
        for count in hash_counter.values()
        if count > 1
    )

    if total_images == 0:

        average_resolution = "0x0"

        average_brightness = 0

    else:

        average_resolution = (
            f"{int(width_sum/total_images)}x"
            f"{int(height_sum/total_images)}"
        )

        average_brightness = float(np.mean(brightness_values))

    return {

        "dataset_name": root.name,

        "total_images": total_images,

        "average_resolution": average_resolution,

        "average_brightness": average_brightness,

        "class_distribution": [
            {
                "class_name": name,
                "count": count,
            }
            for name, count in sorted(class_counter.items())
        ],

        "blurry_images": blurry_images,

        "duplicate_images": duplicate_images,

        "corrupted_images": corrupted_images,

        "invalid_files": invalid_files,

        "empty_class_folders": empty_class_folders,
    }




# ------------------------------------------------------------------
# Vision Inspection
# ------------------------------------------------------------------

def _real_vision_inspection(metadata: Dict[str, Any]) -> Dict[str, Any]:

    dataset_path = metadata["dataset_path"]

    stats = scan_dataset(dataset_path)

    issues = []

    score = 1.0

    total = max(stats["total_images"], 1)

    # -------------------------------------------------------------
    # Blur
    # -------------------------------------------------------------
    if stats["blurry_images"] > 0:

        issues.append(
            {
                "issue_type": "blurry_images",
                "count": stats["blurry_images"],
            }
        )

        score -= min(
            0.25,
            stats["blurry_images"] / total,
        )

    # -------------------------------------------------------------
    # Corrupted Images
    # -------------------------------------------------------------
    if stats["corrupted_images"] > 0:

        issues.append(
            {
                "issue_type": "corrupted_images",
                "count": stats["corrupted_images"],
            }
        )

        score -= min(
            0.25,
            stats["corrupted_images"] / total,
        )

    # -------------------------------------------------------------
    # Duplicate Images
    # -------------------------------------------------------------
    if stats["duplicate_images"] > 0:

        issues.append(
            {
                "issue_type": "duplicate_images",
                "count": stats["duplicate_images"],
            }
        )

        score -= min(
            0.20,
            stats["duplicate_images"] / total,
        )

    # -------------------------------------------------------------
    # Invalid Files
    # -------------------------------------------------------------
    if stats["invalid_files"] > 0:

        issues.append(
            {
                "issue_type": "invalid_files",
                "count": stats["invalid_files"],
            }
        )

        score -= 0.05

    # -------------------------------------------------------------
    # Empty Class Folder
    # -------------------------------------------------------------
    if stats["empty_class_folders"] > 0:

        issues.append(
            {
                "issue_type": "empty_class_folders",
                "count": stats["empty_class_folders"],
            }
        )

        score -= 0.05

    # -------------------------------------------------------------
    # Brightness
    # -------------------------------------------------------------
    brightness = stats["average_brightness"]

    if brightness < 40:

        issues.append(
            {
                "issue_type": "dataset_too_dark",
                "count": 1,
            }
        )

        score -= 0.05

    elif brightness > 220:

        issues.append(
            {
                "issue_type": "dataset_too_bright",
                "count": 1,
            }
        )

        score -= 0.05

    # -------------------------------------------------------------
    # Class imbalance
    # -------------------------------------------------------------
    class_dist = stats["class_distribution"]

    if len(class_dist) > 1:

        counts = [x["count"] for x in class_dist]

        ratio = max(counts) / max(1, min(counts))

        if ratio > 2:

            issues.append(
                {
                    "issue_type": "class_imbalance",
                    "count": int(ratio),
                }
            )

            score -= min(
                0.15,
                (ratio - 2.0) * 0.05,
            )

    score = max(0.0, min(1.0, round(score, 3)))

    metadata.update(stats)

    metadata["vision_quality_score"] = score

    metadata["vision_issues"] = issues

    return {
        "vision_quality_score": score,
        "vision_issues": issues,
    }


# ------------------------------------------------------------------
# Workflow Node
# ------------------------------------------------------------------

def vision_inspect_node(ctx, node_input=None):

    metadata = ctx.state["metadata"]

    result = _real_vision_inspection(metadata)

    ctx.state.update(result)

    return result




def generate_recommendations(
    metadata: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> List[str]:

    recommendations = []

    dataset_name = metadata.get("dataset_name", "Unknown Dataset")
    total_images = metadata.get("total_images", 0)
    resolution = metadata.get("average_resolution", "Unknown")
    score = metadata.get("vision_quality_score", 0.0)

    class_distribution = metadata.get("class_distribution", [])
    num_classes = len(class_distribution)

    issue_types = {issue["issue_type"] for issue in issues}

    # ------------------------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------------------------

    recommendations.append(
        f"The dataset '{dataset_name}' contains {total_images} images across "
        f"{num_classes} classes with an average image resolution of {resolution}."
    )

    recommendations.append(
        f"The overall vision quality score is {score:.2f}/1.00."
    )

    # ------------------------------------------------------------------
    # Quality Assessment
    # ------------------------------------------------------------------

    if not issue_types:
        recommendations.append(
            "No duplicate images, corrupted files, class imbalance, or major image quality issues were detected during automated inspection."
        )

        recommendations.append(
            "The dataset is suitable for computer vision classification experiments and can be used as a strong baseline for CNN or Vision Transformer training."
        )

    # ------------------------------------------------------------------
    # Actionable Recommendations
    # ------------------------------------------------------------------

    if "class_imbalance" in issue_types:
        recommendations.append(
            "Increase the number of samples in minority classes or apply data augmentation to improve class balance."
        )

    if "duplicate_images" in issue_types:
        recommendations.append(
            "Remove duplicated images to improve dataset diversity and reduce overfitting."
        )

    if "blurry_images" in issue_types:
        recommendations.append(
            "Replace blurry images or improve image acquisition quality to increase feature clarity."
        )

    if "corrupted_images" in issue_types:
        recommendations.append(
            "Remove corrupted image files before training to prevent data loading failures."
        )

    if "invalid_image_files" in issue_types:
        recommendations.append(
            "Delete unsupported or invalid image files from the dataset directory."
        )

    if "empty_class_folders" in issue_types:
        recommendations.append(
            "Populate or remove empty class folders to maintain a consistent dataset structure."
        )

    if "dataset_too_dark" in issue_types:
        recommendations.append(
            "Consider collecting brighter images or applying exposure normalization."
        )

    if "dataset_too_bright" in issue_types:
        recommendations.append(
            "Reduce overexposed samples or normalize image brightness."
        )

    recommendations.append(
        "Perform dataset quality inspection before every training cycle to maintain reliable model performance."
    )

    return recommendations


# ------------------------------------------------------------------
# Advisor Input
# ------------------------------------------------------------------

def prepare_advisor_input(ctx, node_input=None):

    metadata = ctx.state["metadata"]

    payload = {
        "dataset_name": metadata["dataset_name"],
        "total_images": metadata["total_images"],
        "average_resolution": metadata["average_resolution"],
        "class_distribution": metadata["class_distribution"],
        "vision_quality_score": metadata["vision_quality_score"],
        "label_issues": metadata["vision_issues"],
        "recommendations": generate_recommendations(
            metadata,
            metadata["vision_issues"],
        ),
    }

    ctx.state["advisor_payload"] = json.dumps(payload, indent=2)

    print("=" * 80)
    print(json.dumps(payload, indent=2))
    print("=" * 80)

    return {
        "advisor_payload": ctx.state["advisor_payload"]
    }


# ------------------------------------------------------------------
# HITL
# ------------------------------------------------------------------

from google.adk.events.request_input import RequestInput

def hitl_gate(ctx, node_input=None):

    print("node_input =", node_input)

    return RequestInput(
        interrupt_id="approval",
        message="""
Dataset analysis is complete.

Please review the analysis.

Type:

yes

to generate the final report

or

no

to reject the report.
""",
        payload=node_input,
        response_schema=str,
    )


# ------------------------------------------------------------------
# Advisor Agent
# ------------------------------------------------------------------

code_executor = UnsafeLocalCodeExecutor()

advisor_agent = LlmAgent(
    name="advisor",
    model="gemini-2.5-flash",
    code_executor=code_executor,
    output_schema=CVReportSchema,
    instruction="""
You are a Computer Vision Dataset Advisor.

The dataset analysis has already been completed.

Here is the COMPLETE analysis JSON:

{advisor_payload}

IMPORTANT RULES

- The JSON above is authoritative.
- NEVER invent any values.
- NEVER invent dataset names.
- NEVER invent image counts.
- NEVER invent classes.
- NEVER estimate statistics.
- Preserve every numeric value exactly.

Your task is ONLY to:

1. Explain the dataset quality.
2. Rewrite the recommendations professionally.
3. Return JSON matching CVReportSchema.

The output must preserve every value from advisor_payload.
""",
)

# ------------------------------------------------------------------
# Report Formatter
# ------------------------------------------------------------------

import json


def report_formatter(ctx, node_input=None):

    if isinstance(node_input, str):
        try:
            report = json.loads(node_input)
        except Exception:
            report = {}
    else:
        report = node_input or {}

    md = f"""
# Vision Dataset QA Report

## Dataset Overview

**Dataset Name:** {report.get("dataset_name","Unknown")}

**Total Images:** {report.get("total_images","Unknown")}

**Average Resolution:** {report.get("average_resolution","Unknown")}

**Vision Quality Score:** {report.get("vision_quality_score","Unknown")}

---

## Class Distribution

"""

    for cls in report.get("class_distribution", []):
        md += f"- **{cls['class_name']}** : {cls['count']} images\n"

    md += "\n---\n\n## Label Issues\n\n"

    issues = report.get("label_issues", [])

    if issues:
        for issue in issues:
            md += f"- {issue}\n"
    else:
        md += "No issues detected.\n"

    md += "\n---\n\n## Recommendations\n\n"

    for rec in report.get("recommendations", []):
        md += f"- {rec}\n"

    print(md)

    ctx.state["markdown_report"] = md

    return {
        "markdown_report": md
    }

# ------------------------------------------------------------------
# Workflow
# ------------------------------------------------------------------

workflow = Workflow(
    name="cv_dataset_advisor",
    description="Computer Vision Dataset Advisor",
    edges=[
        (START, parse_dataset_metadata),
        (parse_dataset_metadata, vision_inspect_node),
        (vision_inspect_node, prepare_advisor_input),
        (prepare_advisor_input, hitl_gate),
        (hitl_gate, advisor_agent),
        (advisor_agent, report_formatter),
    ],
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

root_agent = workflow

__all__ = [
    "root_agent",
    "CVReportSchema",
]