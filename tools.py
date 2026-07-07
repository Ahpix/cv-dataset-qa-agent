import os
import json
import logging
from typing import Any
from google.adk import Context
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def get_text_from_content(content: Any) -> str:
    """Helper to extract raw text string from various input types."""
    if isinstance(content, str):
        return content
    if hasattr(content, "parts") and content.parts:
        return "".join(part.text for part in content.parts if part.text)
    return str(content)


def parse_dataset_metadata(ctx: Context, node_input: Any) -> dict:
    """Input Parsing Node.

    Processes the high-level dataset metadata and stores it in the context state.
    """
    logger.info("Running Input Parsing Node...")
    metadata = {}

    if isinstance(node_input, dict):
        metadata = node_input
    else:
        text = get_text_from_content(node_input)
        try:
            metadata = json.loads(text)
        except Exception:
            # Fallback mock metadata for MVP demonstration
            metadata = {
                "resolution": "1920x1080",
                "brightness_average": "medium-high",
                "class_distribution": {
                    "car": 450,
                    "truck": 120,
                    "pedestrian": 85,
                    "cyclist": 40,
                },
                "total_images": 695,
                "notes": "Failed to parse input as JSON; loaded default capstone dataset metadata.",
                "sample_image_url": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
                "sample_labels": {"boxes": [[100, 150, 400, 500]], "classes": ["car"]},
            }

    # Save to state
    ctx.state["dataset_metadata"] = metadata
    logger.info(f"Metadata stored in context state: {list(metadata.keys())}")
    return {"status": "success", "message": "Dataset metadata parsed successfully."}


def inspect_image_labels(
    image_path_or_bytes: Any, labels_data: dict, model_name: str = "gemini-2.5-flash"
) -> dict:
    """Helper function to perform Gemini Vision inspection or simulation fallback."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        # SIMULATION Mode: Return mock visual QA insights
        logger.info(
            "[SIMULATION MODE] Simulating Gemini Vision visual QA inspection..."
        )
        return {
            "status": "simulation",
            "image_quality": {
                "resolution": "1920x1080 (HD)",
                "contrast": "Good",
                "blurriness": "Very Low",
                "lighting": "Optimal daytime illumination",
            },
            "alignment_qa": {
                "overlapping_boxes_detected": False,
                "missing_annotations_detected": True,
                "misclassifications_detected": False,
                "issues": [
                    "One partially occluded truck in the background is missing a bounding box.",
                    "Class 'car' bounding box aligns perfectly with the primary vehicle (+/- 5px margin).",
                ],
            },
        }

    # LIVE Mode: Execute Gemini Vision API call
    logger.info("[LIVE MODE] Executing live Gemini Vision visual QA inspection...")
    client = genai.Client(api_key=api_key)

    # Process image input into a GenAI Part object
    image_part = None
    if isinstance(image_path_or_bytes, str):
        if image_path_or_bytes.startswith("http://") or image_path_or_bytes.startswith(
            "https://"
        ):
            import requests

            response = requests.get(image_path_or_bytes)
            data = response.content
            mime_type = response.headers.get("Content-Type", "image/jpeg")
            image_part = types.Part.from_bytes(data=data, mime_type=mime_type)
        else:
            with open(image_path_or_bytes, "rb") as f:
                data = f.read()
            mime_type = "image/jpeg"
            if image_path_or_bytes.endswith(".png"):
                mime_type = "image/png"
            image_part = types.Part.from_bytes(data=data, mime_type=mime_type)
    elif isinstance(image_path_or_bytes, bytes):
        image_part = types.Part.from_bytes(
            data=image_path_or_bytes, mime_type="image/jpeg"
        )

    prompt = f"""
    You are an expert Computer Vision QA engineer. Analyze the provided image alongside its annotations:
    Annotations: {labels_data}
    
    Verify the following:
    1. Bounding box alignment: Are the boxes tightly fit around the actual target objects?
    2. QA checks: Are there overlapping boxes, missing annotations, or misclassifications?
    3. Image quality: Assess resolution, contrast, blur, and lighting conditions.
    
    Output a structured JSON object with keys:
    - status (e.g. "live")
    - image_quality (with resolution, contrast, blurriness, lighting)
    - alignment_qa (with overlapping_boxes_detected, missing_annotations_detected, misclassifications_detected, issues)
    """

    contents = [prompt]
    if image_part:
        contents.append(image_part)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(
            f"Gemini Vision API call failed: {e}. Falling back to simulation output."
        )
        return {
            "status": "api_error",
            "message": str(e),
            "alignment_qa": {
                "issues": ["API call failed. Bounding box checks skipped."]
            },
        }


def vision_inspect_node(ctx: Context, node_input: Any) -> dict:
    """Gemini Vision Inspection Node.

    Extracts metadata from context state, calls the vision QA inspection tool,
    and stores results in the context state.
    """
    logger.info("Running Gemini Vision Inspection Node...")
    metadata = ctx.state.get("dataset_metadata", {})

    sample_image = (
        metadata.get("sample_image_url")
        or "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    )
    sample_labels = metadata.get("sample_labels") or {}

    insights = inspect_image_labels(sample_image, sample_labels)
    ctx.state["vision_inspection_results"] = insights
    logger.info("Vision QA inspection completed.")
    return {"status": "success", "message": "Vision QA inspection completed."}


def prepare_advisor_input(ctx: Context, node_input: Any) -> str:
    """Formatting helper node.

    Aggregates metadata and visual QA results from state and shapes them into a single
    consolidated prompt for the final Advisor Agent Node.
    """
    logger.info("Running Prepare Advisor Input Node...")
    metadata = ctx.state.get("dataset_metadata", {})
    vision_results = ctx.state.get("vision_inspection_results", {})

    prompt = f"""
    Please review the following dataset metadata and Gemini Vision quality inspection results.
    
    DATASET METADATA:
    {json.dumps(metadata, indent=2)}
    
    GEMINI VISION QUALITY INSPECTION RESULTS:
    {json.dumps(vision_results, indent=2)}
    
    Analyze the findings carefully. Generate a comprehensive Dataset Advisory Report.
    You must output your response in JSON format. The response will be parsed and validated against the target schema, so make sure to provide all required fields.
    """
    return prompt
