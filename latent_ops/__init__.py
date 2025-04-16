from . import asserts
from . import normalize
from . import functions
from . import ops
from . import images
from . import others

from .base import _NodeMarker
import inspect

NODE_CLASS_MAPPINGS = {}

for module in [asserts, normalize, functions, ops, images, others]:
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if name in NODE_CLASS_MAPPINGS:
            continue
        if name.startswith("_"):
            continue
        if not issubclass(cls, _NodeMarker):
            continue
        NODE_CLASS_MAPPINGS[name] = cls
