import torch
import torchvision
from torchvision import transforms, datasets, models

import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import torch.optim as optim

print(torch.__version__)