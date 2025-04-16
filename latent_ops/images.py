import torch
import torch.nn.functional as tf
from .base import _NodeMarker


class Latent01ToImage(_NodeMarker):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "to_image"

    CATEGORY = "hnmr/latent_ops"

    def to_image(self, latent: dict):
        samples: torch.Tensor = latent["samples"]

        if samples.ndim == 2:
            samples = samples[None, ...]

        if samples.ndim not in (3, 4):
            raise ValueError(f"Expected 3D (C, H, W) or 4D (B, C, H, W) tensor, but got {samples.ndim} dimensions")

        if samples.size(-3) == 1:
            samples = samples.repeat_interleave(3, dim=-3)

        if samples.size(-3) not in (3, 4):
            raise ValueError(
                f"Expected 1, 3, or 4 channels, but got {samples.size(-3)} channels (shape={samples.shape})"
            )

        # breakpoint()
        images = samples.moveaxis(-3, -1)  # (b, c, h, w) -> (b, h, w, c)

        return (images,)


class Latent11ToImage(Latent01ToImage):
    def to_image(self, latent: dict):
        samples = latent["samples"]
        samples = samples * 0.5 + 0.5  # -1..1 -> 0..1
        return super().to_image({"samples": samples})


__all__ = [
    "Latent01ToImage",
    "Latent11ToImage",
]
