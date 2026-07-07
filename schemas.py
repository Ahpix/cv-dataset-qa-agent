from pydantic import BaseModel, Field
from typing import List


class CVReportSchema(BaseModel):
    """Structured report schema for the Computer Vision Dataset Advisor and Label QA Agent."""

    image_quality_status: str = Field(
        ...,
        description="Assessment of the dataset's image quality, including noise, brightness, contrast, resolution, and blur.",
    )
    label_qa_check_results: str = Field(
        ...,
        description="Analysis of the label and bounding box QA check, indicating alignment, overlaps, missing labels, or misclassifications.",
    )
    recommended_data_augmentation_strategies: List[str] = Field(
        ...,
        description="List of specific data augmentation strategies suggested to improve dataset quality, balance, and model training.",
    )
