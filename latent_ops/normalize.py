import torch
import torch.nn.functional as tf
from .utils import input_float, input_int
from .base import _LatentOperation


DEFAULT_EPS = 1e-8


def _float(fn):
    def op(latent: torch.Tensor, **kwargs):
        dtype = latent.dtype
        latent = latent.float()
        result = fn(latent, **kwargs)
        result = result.to(dtype)
        return result

    return op


class LatentOperationNormalizeAlongAxis(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"axis": input_int(default=-1, tooltip="Axis along which to normalize.")},
            # "optional": {"eps": input_float(default=1e-8, tooltip="Epsilon value for numerical stability.")},
        }

    def op(self, axis: int, eps: float | None = None):
        if eps is None:
            eps = DEFAULT_EPS

        @_float
        def normalize(latent: torch.Tensor, **kwargs):
            mean = latent.mean(dim=axis, keepdim=True)
            std = latent.std(dim=axis, keepdim=True)
            return (latent - mean) / (std + eps)

        return (normalize,)


class LatentOperationNormalize(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            # "optional": {"eps": input_float(default=1e-8, tooltip="Epsilon value for numerical stability.")},
        }

    def op(self, eps: float | None = None):
        if eps is None:
            eps = DEFAULT_EPS

        @_float
        def normalize(latent: torch.Tensor, **kwargs):
            mean = latent.mean()
            std = latent.std()
            return (latent - mean) / (std + eps)

        return (normalize,)


class LatentOperationLayerNorm(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            # "optional": {"eps": input_float(default=1e-8, tooltip="Epsilon value for numerical stability.")},
        }

    def op(self, eps: float | None = None):
        if eps is None:
            eps = DEFAULT_EPS

        @_float
        def normalize(latent: torch.Tensor, **kwargs):
            if latent.ndim <= 1:
                return latent
            return tf.layer_norm(latent, latent.shape[1:], eps=eps)

        return (normalize,)


class LatentOperationInstanceNorm(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            # "optional": {"eps": input_float(default=1e-8, tooltip="Epsilon value for numerical stability.")},
        }

    def op(self, eps: float | None = None):
        if eps is None:
            eps = DEFAULT_EPS

        @_float
        def normalize(latent: torch.Tensor, **kwargs):
            if latent.ndim <= 1:
                return latent
            return tf.instance_norm(latent, eps=eps)

        return (normalize,)


class LatentOperationNormalizeMinMax(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            # "optional": {"eps": input_float(default=1e-8, tooltip="Epsilon value for numerical stability.")},
        }

    def op(self, eps: float | None = None):
        if eps is None:
            eps = DEFAULT_EPS

        @_float
        def normalize(latent: torch.Tensor, **kwargs):
            min_val = latent.min()
            max_val = latent.max()
            return (latent - min_val) / (max_val - min_val + eps)

        return (normalize,)


class LatentOperationNormalizePercentile(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"percentile": input_float(default=0.99, min=0.0, max=1.0)},
            # "optional": {"eps": input_float(default=1e-8, tooltip="Epsilon value for numerical stability.")},
        }

    def op(self, percentile: float, eps: float | None = None):
        if eps is None:
            eps = DEFAULT_EPS

        @_float
        def normalize(latent: torch.Tensor, **kwargs):
            v1 = torch.quantile(latent, percentile)
            v2 = torch.quantile(latent, 1 - percentile)
            lo, hi = torch.min(v1, v2), torch.max(v1, v2)
            return (latent - lo) / (hi - lo + eps)

        return (normalize,)


__all__ = [
    "LatentOperationNormalizeAlongAxis",
    "LatentOperationNormalize",
    "LatentOperationLayerNorm",
    "LatentOperationInstanceNorm",
    "LatentOperationNormalizeMinMax",
    "LatentOperationNormalizePercentile",
]
