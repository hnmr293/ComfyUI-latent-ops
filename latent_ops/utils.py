from typing import Literal, Any


def parse_shape(shape: str) -> tuple[int, ...]:
    dims = shape.strip("()").split(",")
    return tuple(int(n.strip()) for n in dims)


def input_float(
    *,
    default: float = 0.0,
    min: float = -10000.0,
    max: float = 10000.0,
    step: float = 0.0001,
    **kwargs,
) -> tuple[Literal["FLOAT"], dict[str, Any]]:
    d: dict = {
        "default": default,
        "min": min,
        "max": max,
        "step": step,
        **kwargs,
    }

    return ("FLOAT", d)


def input_int(
    *,
    default: int = 0,
    min: int = -10000,
    max: int = 10000,
    **kwargs,
) -> tuple[Literal["INT"], dict[str, Any]]:
    d: dict = {
        "default": default,
        "min": min,
        "max": max,
        **kwargs,
    }

    return ("INT", d)
