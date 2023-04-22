import torch
import torch.nn as nn
from torch.nn import functional as F,CrossEntropyLoss
from torch import optim
from torch.nn import LSTM,Linear
import math
import copy
from typing import Callable, Any, Optional, Tuple, List

class windowsize_statistics_model(nn.Module):
    def __init__(self,Api_class_nums, windowsize,*args, **kwargs) -> None:
        super(windowsize_statistics_model,self).__init__()
        self.Linear=nn.Linear(Api_class_nums,windowsize, **kwargs)
        self.classlinear=nn.Linear(windowsize,Api_class_nums, **kwargs)

    def forward(self,statistics_windowsizedata,predict=False):
        #statistics windowsizedata_api      (need transform windowsizeapi to statistics sequence)
        hidden1=self.Linear(statistics_windowsizedata)
        hidden2=self.classlinear(hidden1)
        if predict:
            return F.softmax(hidden2)
        else:
            return hidden2
