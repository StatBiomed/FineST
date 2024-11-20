# add shortcuts to the package's first level

from . import datasets
# from .main import *
from .version import __version__

from .loadData import *
from .processData import *
from .model import *
from .train import *
from .inference import *
from .evaluation import * 
from .downloadData import * 
# from .HIPT_image_feature_extract import * 
from .SparseAEH import *
from .SpatialDM import *
from .plottings import *
from .cropping import *

