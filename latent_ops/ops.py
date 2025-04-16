import torch
from .utils import parse_shape, input_float, input_int
from .base import _LatentOperation, _NodeMarker


class LatentOperationReshape(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"shape": ("STRING", {"tooltip": "Example:\n(1, 2, 3)\n1, 2, 3, 4"})},
            "optional": {"contiguous": ("BOOLEAN", {"default": False, "tooltip": "Whether to reshape continuously."})},
        }

    def op(self, shape: str, contiguous: bool | None = None):
        if contiguous is None:
            contiguous = False

        new_shape = parse_shape(shape)

        def reshape(latent: torch.Tensor, **kwargs):
            new_latent = latent.reshape(new_shape)
            if contiguous:
                new_latent = new_latent.contiguous()
            return new_latent

        return (reshape,)


class LatentOperationSlice(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "axis": input_int(default=-1, tooltip="Axis to slice along."),
                "start": input_int(default=0, tooltip="Start index for slicing."),
                "end": input_int(default=0, tooltip="End index for slicing."),
                "step": input_int(default=1, min=0, tooltip="Step size for slicing."),
            },
        }

    def op(self, axis: int, start: int, end: int, step: int):
        start_ = start if start != 0 else None
        end_ = end if end != 0 else None
        step_ = step if 1 < step else None

        def slice_(latent: torch.Tensor, **kwargs):
            slices = [slice(None)] * latent.ndim
            slices[axis] = slice(start_, end_, step_)
            return latent[tuple(slices)]

        return (slice_,)


class LatentOperationRoll(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shift": input_int(default=0, tooltip="Number of positions to shift."),
                "axis": input_int(default=-1, tooltip="Axis to roll along."),
            },
        }

    def op(self, shift: int, axis: int):
        def roll(latent: torch.Tensor, **kwargs):
            return latent.roll(shift, dims=axis)

        return (roll,)


class LatentOperationAddBroadcast(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": input_float()}}

    def op(self, value: float):
        def add(latent: torch.Tensor, **kwargs):
            return latent + value

        return (add,)


class LatentOperationMulBroadcast(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": input_float(default=1.0)}}

    def op(self, value: float):
        def mul(latent: torch.Tensor, **kwargs):
            return latent * value

        return (mul,)


class LatentOperationFill(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": input_float()}}

    def op(self, value: float):
        def fill(latent: torch.Tensor, **kwargs):
            return torch.ones_like(latent) * value

        return (fill,)


class LatentOperationAdd(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("LATENT",)},
        }

    def op(self, value: torch.Tensor):
        def add(latent: torch.Tensor, **kwargs):
            return latent + value

        return (add,)


class LatentOperationMul(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("LATENT",)},
        }

    def op(self, value: torch.Tensor):
        def mul(latent: torch.Tensor, **kwargs):
            return latent * value

        return (mul,)


class LatentOperationClamp(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min": input_float(default=0.0, tooltip="Minimum value to clamp to."),
                "max": input_float(default=1.0, tooltip="Maximum value to clamp to."),
            },
        }

    def op(self, min: float, max: float):
        def clamp(latent: torch.Tensor, **kwargs):
            return latent.clamp(min=min, max=max)

        return (clamp,)


class LatentOperationClampMin(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"min": input_float(default=0.0, tooltip="Minimum value to clamp to.")}}

    def op(self, min: float):
        def clamp_min(latent: torch.Tensor, **kwargs):
            return latent.clamp(min=min)

        return (clamp_min,)


class LatentOperationClampMax(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"max": input_float(default=1.0, tooltip="Maximum value to clamp to.")}}

    def op(self, max: float):
        def clamp_max(latent: torch.Tensor, **kwargs):
            return latent.clamp(max=max)

        return (clamp_max,)


class LatentOperationApplyCFG(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"scale": input_float(default=1.0, tooltip="result = (1-scale) * uncond + scale * cond")},
        }

    def op(self, scale: float):
        def split_(latent: torch.Tensor, **kwargs):
            uncond, cond = latent.chunk(2, dim=0)
            return uncond + scale * (cond - uncond)

        return (split_,)


class LatentOperationSplitCFG(_NodeMarker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"latent": ("LATENT",)},
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("uncond", "cond")

    FUNCTION = "split"

    CATEGORY = "hnmr/latent_ops"

    def split(self, latent: dict):
        uncond, cond = latent["samples"].chunk(2, dim=0)
        return ({"samples": uncond}, {"samples": cond})


class LatentOperationInterpolate(_NodeMarker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "alpha": input_float(default=0.5, tooltip="z = (1-alpha) * a + alpha * b"),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    FUNCTION = "interpolate"

    CATEGORY = "hnmr/latent_ops"

    def interpolate(self, latent_a: dict, latent_b: dict, alpha: float):
        samples_a = latent_a["samples"]
        samples_b = latent_b["samples"]
        return ({"samples": (1 - alpha) * samples_a + alpha * samples_b},)


__all__ = [
    "LatentOperationReshape",
    "LatentOperationSlice",
    "LatentOperationRoll",
    "LatentOperationAddBroadcast",
    "LatentOperationMulBroadcast",
    "LatentOperationFill",
    "LatentOperationAdd",
    "LatentOperationMul",
    "LatentOperationClamp",
    "LatentOperationClampMin",
    "LatentOperationClampMax",
    "LatentOperationApplyCFG",
    "LatentOperationSplitCFG",
]
