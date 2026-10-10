"""FürElise (Wang et al., SIGGRAPH Asia 2024; CC BY-NC 4.0) motion reader.

The dataset ships Python pickles; pickle can run code on load, so this reader only lets numpy arrays, dtypes and
plain containers through (anything else raises)."""
import io, pickle

import numpy as np

_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("collections", "OrderedDict"), ("builtins", "dict"),
            ("builtins", "list"), ("builtins", "tuple")}


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_pickle(path):
    with open(path, "rb") as f:
        return _Safe(f).load()


def motion(piece_dir):
    """{'left'|'right': {'joints': Nx21x3, ...}} for one piece; FPS 59.94."""
    return load_pickle(f"{piece_dir}/motion.pkl")
