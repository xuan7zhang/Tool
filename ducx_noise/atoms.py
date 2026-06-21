"""Atom tools for the compositional-environment experiment.

These split the **real** TorchXRayVision DenseNet used by ``chest_xray_classifier``
into two BaseTools that share the same weights (no mock, no extra download):

* ``CXREncoderTool`` (``cxr_encoder``): image -> ``features2(x)`` -> 1024-d pooled
  embedding. The embedding is a raw float vector -- a **non-semantic** intermediate
  the driver model cannot faithfully transcribe through a JSON tool-call argument.
* ``CXREmbeddingClassifierTool`` (``cxr_embedding_classifier``): embedding ->
  ``classifier`` + sigmoid + ``op_norm`` -> 18 named pathology probabilities (semantic).

Composing them (``cxr_embedding_classifier ∘ cxr_encoder``) reproduces the original
classifier exactly (see the parity test), but routes the embedding **in memory** so
the model never sees it. That is the basis of the macro-tool experiment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv
from torchxrayvision.models import op_norm
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# 18 pathologies the densenet121-res224-all head predicts (xrv default order).
PATHOLOGIES = list(xrv.datasets.default_pathologies)
EMBEDDING_DIM = 1024


# --------------------------------------------------------------------------
# Shared backbone
# --------------------------------------------------------------------------
def build_densenet(model_name: str = "densenet121-res224-all", device: str = "cuda"):
    model = xrv.models.DenseNet(weights=model_name)
    model.eval()
    dev = torch.device(device) if device else torch.device("cpu")
    return model.to(dev), dev


def _preprocess(image_path: str, device: torch.device) -> torch.Tensor:
    """Identical preprocessing to ChestXRayClassifierTool, for exact parity."""
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    if len(img.shape) > 2:
        img = img[:, :, 0]
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)
    return img.to(device)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class _ImageInput(BaseModel):
    image_path: str = Field(..., description="Path to the chest X-ray image (JPG/PNG).")


class _EmbeddingInput(BaseModel):
    embedding: List[float] = Field(
        ...,
        description=(
            f"A {EMBEDDING_DIM}-dimensional CXR feature embedding produced by cxr_encoder "
            "(raw float vector)."
        ),
    )


# --------------------------------------------------------------------------
# t_a: encoder (image -> embedding)
# --------------------------------------------------------------------------
class CXREncoderTool(BaseTool):
    """Real DenseNet encoder: image -> 1024-d non-semantic embedding."""

    name: str = "cxr_encoder"
    description: str = (
        "Encodes a chest X-ray image into a 1024-dimensional numerical feature embedding "
        "(a raw float vector, not a diagnosis). The embedding must be fed to "
        "cxr_embedding_classifier to obtain pathology predictions."
    )
    args_schema: Type[BaseModel] = _ImageInput
    model: Any = None
    device: Any = None

    def _run(self, image_path: str, run_manager: Optional[Any] = None) -> Tuple[Dict[str, Any], Dict]:
        try:
            img = _preprocess(image_path, self.device)
            with torch.inference_mode():
                emb = self.model.features2(img)  # [1, 1024]
            vec = emb.squeeze(0).cpu().float().tolist()
            return (
                {"embedding": vec, "dim": len(vec)},
                {"image_path": image_path, "analysis_status": "completed"},
            )
        except Exception as exc:  # mirror existing tools' soft-failure convention
            return {"error": str(exc)}, {"image_path": image_path, "analysis_status": "failed"}

    async def _arun(self, image_path: str, run_manager: Optional[Any] = None):
        return self._run(image_path)


# --------------------------------------------------------------------------
# t_b: head (embedding -> disease probabilities)
# --------------------------------------------------------------------------
class CXREmbeddingClassifierTool(BaseTool):
    """Real DenseNet head: 1024-d embedding -> 18 named pathology probabilities."""

    name: str = "cxr_embedding_classifier"
    description: str = (
        "Classifies a 1024-dimensional CXR feature embedding (from cxr_encoder) into "
        "probabilities for 18 pathologies (Atelectasis, Cardiomegaly, Effusion, Pneumonia, "
        "Nodule, Mass, ...). Input must be the exact embedding vector."
    )
    args_schema: Type[BaseModel] = _EmbeddingInput
    model: Any = None
    device: Any = None

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        """Replicate DenseNet.forward's tail after features2()."""
        out = self.model.classifier(features)
        if getattr(self.model, "apply_sigmoid", False):
            out = torch.sigmoid(out)
        if getattr(self.model, "op_threshs", None) is not None:
            out = torch.sigmoid(out)
            out = op_norm(out, self.model.op_threshs)
        return out

    def _run(self, embedding: List[float], run_manager: Optional[Any] = None) -> Tuple[Dict[str, Any], Dict]:
        try:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != EMBEDDING_DIM:
                return (
                    {"error": f"embedding must be a {EMBEDDING_DIM}-d vector, got shape {vec.shape}"},
                    {"analysis_status": "failed"},
                )
            features = torch.from_numpy(vec).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                preds = self._classify(features).cpu()[0]
            output = dict(zip(PATHOLOGIES, preds.numpy().tolist()))
            return output, {"analysis_status": "completed"}
        except Exception as exc:
            return {"error": str(exc)}, {"analysis_status": "failed"}

    async def _arun(self, embedding: List[float], run_manager: Optional[Any] = None):
        return self._run(embedding)


# --------------------------------------------------------------------------
# Factory + atom registry (so the compose transform can build atoms by name)
# --------------------------------------------------------------------------
def build_cxr_atoms(
    device: str = "cuda", model_name: str = "densenet121-res224-all"
) -> Dict[str, BaseTool]:
    """Build encoder + head sharing one real DenseNet instance."""
    model, dev = build_densenet(model_name, device)
    return {
        "cxr_encoder": CXREncoderTool(model=model, device=dev),
        "cxr_embedding_classifier": CXREmbeddingClassifierTool(model=model, device=dev),
    }


# name -> builder that returns a dict of {atom_name: tool} (shared backends grouped)
ATOM_GROUPS = {
    "cxr_densenet": build_cxr_atoms,
}
