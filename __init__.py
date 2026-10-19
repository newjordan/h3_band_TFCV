"""Band Aid: ComfyUI entry point when the whole repo sits in ComfyUI/custom_nodes (git clone or Comfy Registry).

The nodes live in comfy_nodes/; they import the repo's blockout/ and h3band/ code, which is why the package is the
whole repo and not one folder.
"""
from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
