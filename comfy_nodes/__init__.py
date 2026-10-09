"""h3band ComfyUI nodes. Install by linking this folder into ComfyUI/custom_nodes (see the README)."""
import logging

from .h3_band_drums import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from . import h3_band_zone_latent
except ImportError as e:
    # the drum nodes work on any ComfyUI; the zone latent needs the MiniMax H3 nodes
    logging.warning("[h3band] H3 Band Zone Latent not loaded (needs a ComfyUI with MiniMax H3): %s", e)
else:
    NODE_CLASS_MAPPINGS.update(h3_band_zone_latent.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(h3_band_zone_latent.NODE_DISPLAY_NAME_MAPPINGS)
