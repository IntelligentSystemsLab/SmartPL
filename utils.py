import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Deprecated
def equal_space(data_dir,name,save_dir,interval=500):
    file_name = name + '.csv'
    data_path = os.path.join(data_dir,file_name)
    DF = pd.read_csv(data_path)
    y0 = DF.iloc[:,2].values.tolist()
    x = range(0,interval*len(y0))
    xxx = range(0,len(y0))
    xx = DF.iloc[:,1].values.tolist()
    y = []
    for j in range(0,len(y0)):
        t = [v for v in xxx if (j*interval <= xx[v] and  xx[v]<(j+1)*interval)]
        yt = []
        for k in t:
            yt.append(y0[k])
        y.append(np.mean(np.array(yt)))

    data = {'Step': x[:200*interval:interval], 'Value': y[:200]}
    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir,file_name)
    df.to_csv(save_path, index=False)
    print('Processing over!')

# data soomth
def smooth(data, sm=1):
    if sm > 1:
        # for d in data:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        # smooth_data.append(d)
        return d
    return data

#  pack all data in a dir into dataframe
def get_line_data(data_dir):
    data = []
    X = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):  
            file_path = os.path.join(data_dir, filename)  
        df = pd.read_csv(file_path)
        # df = constant_interval(df)
        df['Value'] = smooth(df['Value'],10)
        data.append(df)
    return pd.concat(data)
