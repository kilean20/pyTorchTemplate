from __future__ import print_function
import sys
import os
from copy import deepcopy as copy

# tmp = copy(sys.path)
# sys.path.clear()
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)


from torch_templates import *
import torch_template_plot as plot


sys.path.pop()