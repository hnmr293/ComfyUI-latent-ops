def parse_shape(shape: str) -> tuple[int, ...]:
    dims = shape.strip("()").split(",")
    return tuple(int(n.strip()) for n in dims)
