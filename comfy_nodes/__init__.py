"""Band Aid ComfyUI nodes (h3_band_TFCV).

Install: clone the repo into ComfyUI/custom_nodes (the repo's top-level __init__.py re-exports these mappings), or link
this folder into ComfyUI/custom_nodes. See the README.

Each module loads on its own, so one missing dependency only drops that module's nodes:
  h3_blockout_lock     Blockout Reference Edges, Blockout Lock Check   (any ComfyUI; opencv-python when the nodes run)
  h3_band_zone_latent  H3 Band Zone Latent                             (a ComfyUI with MiniMax H3)
  h3_band_drums        Drum Events, Drum Hits (MIDI), Drum Blockout    (soundfile, the repo's blockout/ and h3band/,
                                                                         Blender when the blockout node runs)
"""
import importlib
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULES = {
    "h3_blockout_lock": "needs numpy and torch",
    "h3_band_zone_latent": "needs a ComfyUI with MiniMax H3",
    "h3_band_drums": "needs soundfile and the repo's blockout/ and h3band/ folders (link this folder, or set H3B_REPO)",
}

for _name, _needs in _MODULES.items():
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
    except Exception as e:  # noqa: BLE001  (one module failing must not take the others down)
        logging.warning("[Band Aid] %s not loaded (%s): %s", _name, _needs, e)
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

logging.info("[Band Aid] %d nodes loaded", len(NODE_CLASS_MAPPINGS))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
