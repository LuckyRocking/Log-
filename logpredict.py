

from keras.models import load_model
import pandas as pd
import numpy as np
import os


#对log日志进行预测正常或非正常
def logpredict(dirtestlog,model,threshold_rate=0.05):
   #对feature每一行数据进行预测，若对应真实标签的概率小于阈值threshold_rate则定位正常与不正常，0为正常，1为不正常。
   #输入dirtestlog为滑动窗口搜集到的数据（为csv格式），model为保存的模型，threahhold为阈值，可以调整。
   ###############################
   #存在边缘缺陷，类似于Cv的滑动窗口,前100个数据没有检测异常，有待补充。
    testlog_DataFrame=pd.read_csv(dirtestlog)

    ############得到特征和标签的名字
    feature_name=testlog_DataFrame.columns[:-1]
    label_name=testlog_DataFrame.columns[-1]
    
    #####划分测试特征数据和标签数据
    feature_data=testlog_DataFrame[feature_name]
    label_data=testlog_DataFrame[label_name]

    ###进行预测
    loc_predict=[]
    abnormalloc=[]
    log_judge='this log is normal'
    for sample_index in range(feature_data.shape[0]):
        true_predictba=model.predict(feature_data.iloc[sample_index].values.reshape(1,-1,))[0][label_data[sample_index]]
        if true_predictba<threshold_rate:
            loc_predict.append(1)
            abnormalloc.append(sample_index+101)
            log_judge='this log is abnormal'
            print('第'+str(sample_index+101)+'api'+'出现异常')
        else:
            loc_predict.append(0)
            print('第'+str(sample_index+101)+'api'+'正常')
    
    return loc_predict,abnormalloc,log_judge

if __name__=='__main__':
   
   model=load_model(r'E:\logdiff Data\测试集及深度学习model\logdiffmodel.h5') #######修改路径
   dir_testsample_path=''             #为测试样本数据集包含5个csv文件的文件夹路径
   testsample_path=[os.join(dir_testsample_path,csvname)   for csvname in os.listdir(dir_testsample_path)]   #5个复现log日志数据路径
   alltestsample_predictlis=[]
   abnormalloc_lis=[]
   for log_path  in testsample_path:
        loc_predict,abnormalloc,log_judge=logpredict(log_path,model,threshold_rate=0.05)      
        alltestsample_predictlis.append(loc_predict)
        abnormalloc_lis.append(abnormalloc)
        print(log_judge)
   
#输出结果alltestsample_predict为大列表，里面每个小列表为每个log日志预测结果，其中0为正常，1为不正常。
#abnormalloc_lis为大列表，里面小列表为每个log日志预测不正常的apiindex值。
#每次打印log_judge看下这个log日志是不是有问题，若'this log is normal'为正常，'this log is abnormal'为不正常