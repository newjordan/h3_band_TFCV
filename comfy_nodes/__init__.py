"""h3band ComfyUI nodes. Install by linking this folder into ComfyUI/custom_nodes (see the README)."""
import logging

from .h3_band_drums import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from . import h3_band_zone_latent
except ImportError as e:
    # the drum nodes work on any ComfyUI; the zone latent needs MiniMax H3 + per-token AV noise masks (PR #15375)
    logging.warning("[h3band] H3 Band Zone Latent not loaded (needs ComfyUI with MiniMax H3 and PR #15375): %s", e)
else:
    NODE_CLASS_MAPPINGS.update(h3_band_zone_latent.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(h3_band_zone_latent.NODE_DISPLAY_NAME_MAPPINGS)
