import csv
import os
import pandas as pd
import numpy as np


def getapifunindex(api_sequence_path,apifunindex_path,dirFilApiFuns):
    '''api_sequence为numpy数据（bert或者降维处理后的），dirFilApiFuns为文件输出路径'''
    if '.np' in api_sequence_path:
        np_api_sequence=np.load(api_sequence_path)
        dims=len(np_api_sequence[0])
    #得到dirapifunindex的apiname
        apiname=pd.read_csv(apifunindex_path)['ApiFunName']
    #整合apiname和np_data于DataFrame
        pd.DataFrame(np_api_sequence,index=apiname).to_csv(dirFilApiFuns)
    elif '.csv' in api_sequence_path:
        df_api_sequence=pd.read_csv(api_sequence_path)
        dims=len(df_api_sequence.iloc[0])
        df_api_sequence.index=pd.read_csv(apifunindex_path)['ApiFunName'] #得到dirapifunindex的apiname
        df_api_sequence.to_csv(dirFilApiFuns)      
    return  dims

#有了api嵌入向量文件,对log日志进行提模板
def apiFunFil2Dct(dirFilApiFuns):
    if(not os.path.isfile(dirFilApiFuns)):
        print("The api function file path are incorrect")
        return
    
    dfApifun = pd.read_csv(dirFilApiFuns) 

    word_sequence_columns=dfApifun.columns[1:]

    dctApiFun = dict(zip(dfApifun['ApiFunName'],dfApifun[word_sequence_columns].values.tolist()))      
    
    return dctApiFun

#输出
def lst2csv(lstCsv,dirFilCsv):    
    pFile = open(dirFilCsv, "w", encoding = "utf-8", newline='')
    csvWriter = csv.writer(pFile)
    for row in lstCsv:
        csvWriter.writerow(row)    
    pFile.close()
    return len(lstCsv)


def filOptimize(dctApiFun, dirFilApiLog, dirFilOptimized,dims):  
    if(not os.path.isfile(dirFilApiLog)):
        print("The input parameters are incorrect")
        return
    
    pCsv = open(dirFilApiLog,'r',encoding = "utf-8")
    csvReader = csv.reader(pCsv)
    csvHeader = next(csvReader)
    lstOptimized = []

    lstOptimized.append([csvHeader[4]]+['Apisequencedim'+str(i) for i in range(dims)])
    for row in csvReader:
        if((row[0]=="API" or row[0]=="SAPI") and row[3].endswith(".dll") and row[7] != ''):
            
            # 1、剥离函数名称
            sFunName = row[4][0:row[4].find('(')]
            iFunNamHead = sFunName.rfind(' ')
            sFunName = sFunName[iFunNamHead + 1:]
            
            
            # 3、添加函数索引(函数ID)列
            # iFunIndex = dctApiFun[sItem]
            iFunIndex = dctApiFun.get(sFunName)          # 最好用get函数而不用下标索引，不会发生异常！
            if iFunIndex == None:           
                iFunIndex = [0 for i in range(dims)]
                # print(iRowCnt,'——',sItem)   
            
            # 4、保留有用的列，摈弃其它列
            lstOptimized.append([sFunName]+iFunIndex)
    pCsv.close()

    lst2csv(lstOptimized, dirFilOptimized)        
    return len(lstOptimized)


def dirLogOptimize(dirFilApiFuns, dirApiLog, dirOptimized,dims):
    if not os.path.exists(dirOptimized):
        os.mkdir(dirOptimized)              # 如果文件夹不存在就创建它
    
    print()
    print(dirApiLog, '--->', dirOptimized)
    
    dctApiFun = apiFunFil2Dct(dirFilApiFuns)
    
    # print()
    lstFiles = os.listdir(dirApiLog)
    iFilCnt = 0
    for file in lstFiles:
        dirFilApiLog = os.path.join(dirApiLog, file)
        if os.path.isfile(dirFilApiLog):
            (filTitApiLogExt, filExtApiLogExt) = os.path.splitext(file)
            dirFilOptimized = os.path.join(dirOptimized, filTitApiLogExt + '_optimized' + filExtApiLogExt)
            print(dirFilApiLog, '--->', dirFilOptimized)		# 打印输出每个文件全路径
            iTabRows = filOptimize(dctApiFun, dirFilApiLog, dirFilOptimized,dims)
            # print('Records = ', iTabRows)		# 打印输出每个文件(夹)全路径
            iFilCnt += 1
    
    print()
    print('A total of ', iFilCnt, ' files were processed.')
    
    return iFilCnt

def getlogoptimized(api_sequence_path,apifunindex_path,dirApiLog,dirFilApiFuns,dirOptimized):
    '''api_sequence_path为api向量路径，apifunindex_path为api_index路径，dirFilApiFuns为向量模板输出路径，dirApilog为apilog文件夹'''
    dims=getapifunindex(api_sequence_path,apifunindex_path,dirFilApiFuns)
    dirLogOptimize(dirFilApiFuns, dirApiLog, dirOptimized,dims)
    return 


if __name__=='__main__':
    api_sequence_path=r'D:\Backup\Downloads\api_embed_pca_random.np'
    apifunindex_path=r'E:\logdiff Data\apiFunList.csv'
    dirApiLog=r'E:\logdiff Data\two log diff data\gilisoft file not repro'
    dirOptimized=r'E:\logdiff Data\gilisoft file not repro optimized'
    dirFilApiFuns=r'E:\logdiff Data\api_embeding.csv'
    getlogoptimized(api_sequence_path,apifunindex_path,dirApiLog,dirFilApiFuns,dirOptimized)