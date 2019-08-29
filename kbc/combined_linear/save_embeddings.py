import os
import sys
import time
import glob
import tqdm
import torch
import utils
import logging
import argparse
import torch.utils
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch import optim
from typing import Dict
from datasets import Dataset
from architect import Architect
from model_search import Network
from torch.autograd import Variable
from regularizers import N2, N3, Regularizer

model = torch.load('search-EXP-20190823-173036%f/weights.pt')
embeddings = model.embeddings
torch.save(embeddings, 'search-EXP-20190823-173036%f/embeddings.pt')
