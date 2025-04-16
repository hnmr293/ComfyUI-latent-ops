class _NodeMarker:
    pass


class _LatentOperation(_NodeMarker):
    RETURN_TYPES = ("LATENT_OPERATION",)
    RETURN_NAMES = ("op",)

    FUNCTION = "op"
    CATEGORY = "hnmr/latent_ops"
