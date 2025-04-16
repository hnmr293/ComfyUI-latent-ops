import torch
import torch.nn.functional as tf
from .base import _LatentOperation


class LatentOperationSigmoid(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def sigmoid(latent: torch.Tensor, **kwargs):
            return tf.sigmoid(latent * alpha)

        return (sigmoid,)


class LatentOperationHardSigmoid(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def hard_sigmoid(latent: torch.Tensor, **kwargs):
            return tf.hardsigmoid(latent * alpha)

        return (hard_sigmoid,)


class LatentOperationLogistic(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def logistic(latent: torch.Tensor, **kwargs):
            return tf.logsigmoid(latent * alpha)

        return (logistic,)


class LatentOperationTanh(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def tanh(latent: torch.Tensor, **kwargs):
            return tf.tanh(latent * alpha)

        return (tanh,)


class LatentOperationHardTanh(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def hard_tanh(latent: torch.Tensor, **kwargs):
            return tf.hardtanh(latent * alpha)

        return (hard_tanh,)


class LatentOperationSinh(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def sinh(latent: torch.Tensor, **kwargs):
            return torch.sinh(latent * alpha)

        return (sinh,)


class LatentOperationCosh(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def cosh(latent: torch.Tensor, **kwargs):
            return torch.cosh(latent * alpha)

        return (cosh,)


class LatentOperationReLU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def relu(latent: torch.Tensor, **kwargs):
            return tf.relu(latent * alpha)

        return (relu,)


class LatentOperationReLU6(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def relu6(latent: torch.Tensor, **kwargs):
            return tf.relu6(latent * alpha)

        return (relu6,)


class LatentOperationLeakyReLU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
            "optional": {"negative_slope": ("FLOAT", {"default": 0.01})},
        }

    def op(self, alpha: float, negative_slope: float | None = None):
        if negative_slope is None:
            negative_slope = 0.01

        def leaky_relu(latent: torch.Tensor, **kwargs):
            return tf.leaky_relu(latent * alpha, negative_slope=negative_slope)

        return (leaky_relu,)


class LatentOperationELU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def elu(latent: torch.Tensor, **kwargs):
            return tf.elu(latent * alpha)

        return (elu,)


class LatentOperationSELU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def selu(latent: torch.Tensor, **kwargs):
            return tf.selu(latent * alpha)

        return (selu,)


class LatentOperationCELU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def celu(latent: torch.Tensor, **kwargs):
            return tf.celu(latent * alpha)

        return (celu,)


class LatentOperationGELU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def gelu(latent: torch.Tensor, **kwargs):
            return tf.gelu(latent * alpha)

        return (gelu,)


class LatentOperationSiLU(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def silu(latent: torch.Tensor, **kwargs):
            return tf.silu(latent * alpha)

        return (silu,)


class LatentOperationHardSwish(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def hard_swish(latent: torch.Tensor, **kwargs):
            return tf.hardswish(latent * alpha)

        return (hard_swish,)


class LatentOperationMish(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def mish(latent: torch.Tensor, **kwargs):
            return tf.mish(latent * alpha)

        return (mish,)


class LatentOperationSoftplus(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"beta": ("FLOAT", {"default": 1.0})},
        }

    def op(self, beta: float):
        def softplus(latent: torch.Tensor, **kwargs):
            return tf.softplus(latent * beta)

        return (softplus,)


class LatentOperationSoftmax(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"axis": ("INT", {"default": -1})}}

    def op(self, axis: int):
        def softmax(latent: torch.Tensor, **kwargs):
            return tf.softmax(latent, dim=axis)

        return (softmax,)


class LatentOperationSoftmin(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"axis": ("INT", {"default": -1})}}

    def op(self, axis: int):
        def softmin(latent: torch.Tensor, **kwargs):
            return tf.softmin(latent, dim=axis)

        return (softmin,)


class LatentOperationSoftsign(_LatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"alpha": ("FLOAT", {"default": 1.0})},
        }

    def op(self, alpha: float):
        def softsign(latent: torch.Tensor, **kwargs):
            return tf.softsign(latent * alpha)

        return (softsign,)


__all__ = [
    "LatentOperationSigmoid",
    "LatentOperationHardSigmoid",
    "LatentOperationLogistic",
    "LatentOperationTanh",
    "LatentOperationHardTanh",
    "LatentOperationSinh",
    "LatentOperationCosh",
    "LatentOperationReLU",
    "LatentOperationReLU6",
    "LatentOperationLeakyReLU",
    "LatentOperationELU",
    "LatentOperationSELU",
    "LatentOperationCELU",
    "LatentOperationGELU",
    "LatentOperationSiLU",
    "LatentOperationHardSwish",
    "LatentOperationMish",
    "LatentOperationSoftplus",
    "LatentOperationSoftmax",
    "LatentOperationSoftmin",
    "LatentOperationSoftsign",
]
