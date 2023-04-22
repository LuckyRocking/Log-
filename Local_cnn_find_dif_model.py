import torch
import torch.nn as nn
from torch.nn import functional as F,CrossEntropyLoss
from torch import optim
from torch.nn import LSTM,Linear
import math
import copy
from typing import Callable, Any, Optional, Tuple, List



class Conv1dlocal_find_sequence_dif_model(nn.Module):
    def __init__(self, windowsize,Api_dim,Api_classnum,outchannels1,outchannels2,**kwargs) -> None:
        super().__init__()
        #bigger_conv1d to find local_dif
        self.bigger_conv1d=nn.Conv1d(in_channels=Api_dim,out_channels=outchannels1,kernel_size=5,padding=2,bias=True,**kwargs)
        
        self.bn1=nn.BatchNorm1d(num_features=outchannels1,eps=0.001)

        #smaller_cov1d to find local_dif
        self.smaller_conv1d=nn.Conv1d(in_channels=Api_dim,out_channels=outchannels2,kernel_size=3,padding=1,bias=True,**kwargs)

        self.bn2=nn.BatchNorm1d(num_features=outchannels2,eps=0.001)

        self.flatten=nn.Flatten()

        self.classlinear=nn.Linear(windowsize*(outchannels1+outchannels2),Api_classnum)

    def forward(self,window_sizedata,predict=False):
        #faster compute 
        window_sizedata=torch.tensor(window_sizedata) if type(window_sizedata)!=torch.tensor else window_sizedata
        if window_sizedata.dim()==2:
            window_sizedata=window_sizedata.reshape(1,window_sizedata.size(0),window_sizedata.size(1))
        
        #need permuter window_sizedata to (batch_size,channels,api_windowsize)
        window_sizedata=window_sizedata.permute(0,2,1)
            
        bigger_conv=F.relu(self.bn1(self.bigger_conv1d(window_sizedata)))

        smaller_conv=F.relu(self.bn2(self.smaller_conv1d(window_sizedata)))

        all_conv=torch.concat([bigger_conv,smaller_conv],axis=1).permute(0,2,1)
        

        all_conv=self.flatten(all_conv)
        
        if predict:
            return F.softmax(self.classlinear(all_conv))
        else:
            return self.classlinear(all_conv)