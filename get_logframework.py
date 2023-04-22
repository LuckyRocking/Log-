# coding: utf-8
import os
import csv
import time
import pandas as pd



def lst2csv(lstCsv,dirFilCsv):    
    pFile = open(dirFilCsv, "w", encoding = "utf-8", newline='')
    csvWriter = csv.writer(pFile)
    for row in lstCsv:
        csvWriter.writerow(row)    
    pFile.close()
    
    return len(lstCsv)

def apiFunFil2Dct(dirFilApiFuns):
    if(not os.path.isfile(dirFilApiFuns)):
        print("The api function file path are incorrect")
        return
    
    dfApifun = pd.read_csv(dirFilApiFuns)    
    dctApiFun = dict(zip(dfApifun['ApiFunName'], dfApifun['Number']))       # Api函数的索引字典，从〇算起
    
    return dctApiFun

# 功能：优化【ApiLogExt】文件
# 参数：
# dirFilApiLogExt——原始【ApiLogExt】文件
# dirFilOptimized——优化后的log文件
def filOptimize(dctApiFun, dirFilApiLogExt, dirFilOptimized):    
    if(not os.path.isfile(dirFilApiLogExt)):
        print("The input parameters are incorrect")
        return
    
    pCsv = open(dirFilApiLogExt,'r',encoding = "utf-8")
    csvReader = csv.reader(pCsv)
    csvHeader = next(csvReader)
    lstOptimized = []
    
    # 共 4 个字段：【Number】,【result】，【error code】，【Apifunindex】
    lstOptimized.append(('Idh',csvHeader[2],csvHeader[3], 'ApiFunIndex'))

    for row in csvReader:
        if((row[1]=="API" or row[1]=="SAPI") and row[2]!='*' and row[4].endswith(".dll") and row[8] != '' ):
            
            # 1、剥离函数名称
            sFunName = row[5][0:row[5].find('(')]
            iFunNamHead = sFunName.rfind(' ')
            sFunName = sFunName[iFunNamHead + 1:]
            
            
            
            # 2、添加函数索引(函数ID)列
            # iFunIndex = dctApiFun[sItem]
            iFunIndex = dctApiFun.get(sFunName)          # 最好用get函数而不用下标索引，不会发生异常！
            if iFunIndex == None:           # 如果字典中不存在此词条，则置索引为 -1
                iFunIndex = -1
                # print(iRowCnt,'——',sItem)   
            
            # 3、优化Error_code('0'为正常，‘1’为不正常，原始日志中带有errorcode)
            error_label=0 if row[3]=='0' else 1

            #4、优化Result结果
            result_label = 0 if row[2]=='OK' else 1
            
            # 4、保留有用的列，摈弃其它列
            lstOptimized.append((row[0],result_label, error_label, iFunIndex))
    pCsv.close()

    lst2csv(lstOptimized, dirFilOptimized)        
    return len(lstOptimized)


def dirLogOptimize(dirFilApiFuns, dirApiLogExt, dirOptimized):
    if not os.path.exists(dirOptimized):
        os.mkdir(dirOptimized)              # 如果文件夹不存在就创建它
    
    print()
    print(dirApiLogExt, '--->', dirOptimized)
    
    dctApiFun = apiFunFil2Dct(dirFilApiFuns)
    
    # print()
    lstFiles = os.listdir(dirApiLogExt)
    iFilCnt = 0
    for file in lstFiles:
        dirFilApiLogExt = os.path.join(dirApiLogExt, file)
        if os.path.isfile(dirFilApiLogExt):
            (filTitApiLogExt, filExtApiLogExt) = os.path.splitext(file)
            dirFilOptimized = os.path.join(dirOptimized, filTitApiLogExt + '_optimized' + filExtApiLogExt)
            print(dirFilApiLogExt, '--->', dirFilOptimized)		# 打印输出每个文件(夹)全路径
            iTabRows = filOptimize(dctApiFun, dirFilApiLogExt, dirFilOptimized)
            # print('Records = ', iTabRows)		# 打印输出每个文件(夹)全路径
            iFilCnt += 1
    
    print()
    print('A total of ', iFilCnt, ' files were processed.')
    
    return iFilCnt

if __name__ == '__main__':
    # dirCurrent = os.path.dirname(__file__)      # 当前目录，相对路径，不可靠
    # dirCurrent = os.path.abspath(os.path.dirname(__file__))         # 当前目录，绝对路径，可靠
    # dirFather = os.path.abspath(os.path.dirname(dirCurrent))        # 父目录
    # dirDatSet = os.path.join(dirFather, 'datSet')
    
    # folderApiLogExt = 'apiLogZxl'
    # dirApiLogExt = os.path.join(dirDatSet, folderApiLogExt)
    
    
    # Api 函数字典文件
    dirFilApiFuns = r'E:\logdiff Data\apiFunList.csv'
    
    # 源文件夹：
    # dirApiLogExt = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\apiLogOriginals\\truncated_test'
    # dirApiLogExt = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\apiLogOriginals\\ckpWin11Ext'
    # dirApiLogExt = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\apiLogOriginals\\timExt'
    # dirApiLogExt = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\apiLogOriginals\\zxlExt'
    dirApiLogExt = r'E:\logdiff Data\two log diff data\0x1- CPUSTRES_Win10'
    
    # 生成目标文件夹
    # dirOptimized = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\intermediates\\simplified_test'
    # dirOptimized = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\intermediates\\ckpWin11Optimized'
    # dirOptimized = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\intermediates\\timOptimized'
    # dirOptimized = 'D:\\projects\\pythonProjects\\AI\\apiLogTorch\\datSet\\intermediates\\zxlOptimized'
    dirOptimized = r'E:\logdiff Data\two log diff data\0x1- CPUSTRES_Win10_optimized'
    
    if not os.path.exists(dirOptimized):
        os.mkdir(dirOptimized)                   # 如果文件夹不存在就创建它
    
    timStart = time.time()          # 开始计时
    
    dirLogOptimize(dirFilApiFuns, dirApiLogExt, dirOptimized)
    
    print('Finished.')
    timEnd = time.time()            # 计时结束
    timGap = timEnd - timStart
    print('耗时：', timGap, '秒。')