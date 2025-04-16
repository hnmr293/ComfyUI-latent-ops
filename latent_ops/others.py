import torch
import torch.nn.functional as tf
from .base import _NodeMarker


class GetSigma(_NodeMarker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "index": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("sigma",)

    FUNCTION = "op"

    CATEGORY = "hnmr/latent_ops"

    def op(self, sigmas: torch.Tensor, index: int):
        if sigmas.ndim != 1:
            raise ValueError(f"Expected 1D tensor, but got {sigmas.ndim}D tensor")

        sigma = sigmas[index]

        return (sigma,)


__all__ = [
    "GetSigma",
]
