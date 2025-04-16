import torch

from .utils import parse_shape
from .base import _NodeMarker


class AssertDims(_NodeMarker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "dims": ("INT",),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "first_dim", "last_dim")

    FUNCTION = "op"

    CATEGORY = "hnmr/latent_ops"

    def op(self, latent: dict, dims: int):
        """
        Asserts that the input tensor has the specified number of dimensions.

        Args:
            latent (dict): The input tensor to be checked.
            dims (int): The expected number of dimensions.

        Raises:
            ValueError: If the input tensor does not have the expected number of dimensions.
        """
        # Check if the input tensor has the expected number of dimensions
        samples: torch.Tensor = latent["samples"]
        if samples.ndim != dims:
            raise ValueError(f"Expected {dims} dimensions, but got {samples.ndim}")

        if samples.ndim == 0:
            first = 0
            last = 0
        else:
            first = samples.size(0)
            last = samples.size(-1)

        return latent.copy(), first, last


class AssertShape(_NodeMarker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "shape": ("STRING", {"tooltip": "Example:\n(1, 2, 3)\n1, 2, 3, 4"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "first_dim", "last_dim")

    FUNCTION = "op"

    CATEGORY = "hnmr/latent_ops"

    def op(self, latent: dict, shape: str):
        """
        Asserts that the input tensor has the specified shape.

        Args:
            tensor (dict): The input tensor to be checked.
            shape (str): The expected shape in the format "(dim1, dim2, ...)".

        Raises:
            ValueError: If the input tensor does not have the expected shape.
        """
        # Check if the input tensor has the expected shape
        samples: torch.Tensor = latent["samples"]
        expected_shape = parse_shape(shape)
        if samples.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, but got {samples.shape}")

        if samples.ndim == 0:
            first = 0
            last = 0
        else:
            first = samples.size(0)
            last = samples.size(-1)

        return latent.copy(), first, last


__all__ = [
    "AssertDims",
    "AssertShape",
]
