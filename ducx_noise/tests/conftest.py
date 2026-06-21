"""Lightweight fake tools for noise unit tests (no GPU, no LLM, no real models)."""

from typing import Type

import pytest
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _ImageInput(BaseModel):
    image_path: str = Field(..., description="Path to the radiology image file")


class FakeClassifier(BaseTool):
    name: str = "chest_xray_classifier"
    description: str = "Classifies chest X-ray images for 18 pathologies."
    args_schema: Type[BaseModel] = _ImageInput

    def _run(self, image_path: str, run_manager=None):
        return ({"Pneumonia": 0.7}, {"analysis_status": "completed", "image_path": image_path})


class FakeSegmentation(BaseTool):
    name: str = "chest_xray_segmentation"
    description: str = "Segments anatomical structures in a chest X-ray."
    args_schema: Type[BaseModel] = _ImageInput

    def _run(self, image_path: str, run_manager=None):
        return ({"left_lung": 0.5}, {"analysis_status": "completed"})


@pytest.fixture
def base_tools():
    return [FakeClassifier(), FakeSegmentation()]
