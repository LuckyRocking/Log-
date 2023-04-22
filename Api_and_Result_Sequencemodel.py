import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import LSTM,Linear
from typing import Callable, Any, Optional, Tuple, List

   
class Api_Sequence2Result_Sequence_finddif_model(nn.Module):
    '''Api_Sequence to result_Sequence_to next_Api_model'''
    def __init__(self,Api_dim,Api_class_nums,windows_size,hidden_layers1,hidden_layers2,result_channels,lstm1_units,lstm2_units,lstm3_units,lstm4_units,**kwargs:Any) -> None:
        super(Api_Sequence2Result_Sequence_finddif_model,self).__init__()
        
        #####first_dim=batch_size
        self.lstm1=nn.LSTM(input_size=Api_dim,hidden_size=hidden_layers1,num_layers=lstm1_units,batch_first=True,**kwargs)   

        self.result_class_linear=nn.Linear(hidden_layers1,2,bias=True)     

        self.lstm2=nn.LSTM(input_size=result_channels,hidden_size=hidden_layers2,num_layers=lstm2_units,batch_first=True,**kwargs)

        self.apilstm1=nn.LSTM(input_size=Api_dim,hidden_size=hidden_layers2,num_layers=lstm3_units,batch_first=True,**kwargs)

        self.apilstm2=nn.LSTM(input_size=Api_dim,hidden_size=hidden_layers2,num_layers=lstm4_units,batch_first=True,**kwargs)

        self.Flatten1=nn.Flatten()

        self.Dropout=nn.Dropout(0.1)

        self.ApiLinear=nn.Linear(hidden_layers2*windows_size+(windows_size+(windows_size)//2)*hidden_layers2,Api_class_nums)
    

    def forward(self,window_sizedata,Dropout=True,predict=False):
        #forward 
        #faster compute
        window_sizedata=torch.tensor(window_sizedata) if type(window_sizedata)!=torch.tensor else window_sizedata
        if window_sizedata.dim()==2:
            window_sizedata=window_sizedata.reshape(1,window_sizedata.size(0),window_sizedata.size(1))
        # batch forward 
         #to get output of everystep and the outputdim of everystep is hidden_size     
        api2result_sequence=self.result_class_linear(self.lstm1(window_sizedata)[0])

        #utilize info of api->result_sequence to next lstm
        result_sequence=self.lstm2(api2result_sequence)[0]
         
        #bigger windowsize lstm
        coarse_Apisequence_output=self.apilstm1(window_sizedata)[0]

        #smaller windowsize lstm
        fine_Apisequence_output=self.apilstm2(window_sizedata[:,-(window_sizedata).size(1)//2:,])[0]   
        
        Apisequence=torch.concat([coarse_Apisequence_output,fine_Apisequence_output],axis=1) 
        
        #utilize result_sequence to class
        result_sequence_flatten=self.Flatten1(result_sequence)

        api_sequence_flatten=self.Flatten1(Apisequence)
        
        all_suquence=torch.concat([result_sequence_flatten,api_sequence_flatten],axis=1) 
        
        if Dropout:
        #Dropout 
            allsequence_afdpt=self.Dropout(all_suquence)
        #class
            Api_class=self.ApiLinear(allsequence_afdpt)

        else:
            Api_class=self.ApiLinear(all_suquence)

        if predict:

            return  api2result_sequence,F.softmax(Api_class)
        
        else:

            return  api2result_sequence,Api_class

