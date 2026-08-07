## add shortcuts to the package's first level
from . import datasets
from .version import __version__

# Explicit submodule imports so FineST.model / FineST.traintest keep working
# even when this package defines __getattr__.
from . import loadData
from . import processData
from . import model
from . import traintest
from . import inference
from . import evaluation
from . import SparseAEH
from . import plottings
from . import downstream
from . import utils

from .loadData import *
from .processData import *
from .model import *
from .traintest import *
from .inference import *
from .evaluation import *
from .SparseAEH import *
from .plottings import *
from .downstream import *
from .utils import *

# Lazy API exports (avoid shadowing submodules needed for ``python -m ...``)
_LAZY_API = {
    'image_feature_extraction': ('.image_feature_extraction', 'image_feature_extraction'),
    'step1_FineST_train_infer': ('.step1_FineST_train_infer', 'step1_FineST_train_infer'),
    'spot_interpolation': ('.spot_interpolation', 'spot_interpolation'),
    'nuclei_segmentation': ('.nuclei_segmentation', 'nuclei_segmentation'),
    'step2_high_resolution_imputation': (
        '.step2_High_resolution_impute',
        'step2_high_resolution_imputation',
    ),
    'tutorial_path_presets': ('.paths', 'tutorial_path_presets'),
    'visiumhd_path_presets': ('.paths', 'visiumhd_path_presets'),
}

_FALLBACK_MODULES = (
    'model',
    'traintest',
    'processData',
    'inference',
    'evaluation',
    'loadData',
    'utils',
    'plottings',
    'downstream',
)


def __getattr__(name):
    import importlib

    if name in _LAZY_API:
        mod_name, attr = _LAZY_API[name]
        mod = importlib.import_module(mod_name, __name__)
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj

    for mod_name in _FALLBACK_MODULES:
        mod = importlib.import_module(f'.{mod_name}', __name__)
        if hasattr(mod, name):
            obj = getattr(mod, name)
            globals()[name] = obj
            return obj

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_FALLBACK_MODULES) | set(_LAZY_API))
