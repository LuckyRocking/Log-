import torch
import torch.nn as nn
from torch.nn import functional as F,CrossEntropyLoss
from torch import optim
from torch.nn import LSTM,Linear
import math
import copy
from typing import Callable, Any, Optional, Tuple, List




class Lstmlocal_find_sequence_dif_model(nn.Module):
    def __init__(self, Api_dim,Api_class,hidden_layers,bigger_window_size=10,smaller_window_size=5, **kwargs) -> None:
        super(Lstmlocal_find_sequence_dif_model,self).__init__()
        #define windows size 
        self.bigger_window_size=bigger_window_size
        
        self.smaller_window_size=smaller_window_size

        #window_size is 10
        self.apilstm1=nn.LSTM(input_size=Api_dim,hidden_size=hidden_layers,batch_first=True,**kwargs)
        #window_size is 5
        self.apilstm2=nn.LSTM(input_size=Api_dim,hidden_size=hidden_layers,batch_first=True,**kwargs)
        
        self.Dropout=nn.Dropout(0.1)
        
        self.Flatten=nn.Flatten()

        self.classlinear=nn.Linear((bigger_window_size+smaller_window_size)*hidden_layers,Api_class)

    def forward(self,window_sizedata,Dropout=True,predict=False):
        #faster compute
        window_sizedata=torch.tensor(window_sizedata) if type(window_sizedata)!=torch.tensor else window_sizedata
        if window_sizedata.dim()==2:
            window_sizedata=window_sizedata.reshape(1,window_sizedata.size(0),window_sizedata.size(1))
        
        
        bigger_window_size=self.bigger_window_size

        smaller_window_size=self.smaller_window_size

        biggerlstmoutput=self.apilstm1(window_sizedata[:,-bigger_window_size:,])[0]

        smallerlstmouput=self.apilstm2(window_sizedata[:,-smaller_window_size:,])[0]

        if Dropout:

            biggerlstmoutput=self.Dropout(biggerlstmoutput)

            smallerlstmouput=self.Dropout(smallerlstmouput)
        
        api_sequence=torch.concat([biggerlstmoutput,smallerlstmouput],axis=1)


        flatten_api_sequence=self.Flatten(api_sequence)

        if predict:
            return  F.softmax(self.classlinear(flatten_api_sequence))
        else:
            return self.classlinear(flatten_api_sequence)