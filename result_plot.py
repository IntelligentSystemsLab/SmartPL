import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

labels = []


def smooth(data, sm=1):
    if sm > 1:
        # for d in data:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        # smooth_data.append(d)
        return d
    return data


def get_line_data(data_dir):
    data = []
    X = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):  # 检查是否为 CSV 文件
            file_path = os.path.join(data_dir, filename)  # 构建完整文件路径
        df = pd.read_csv(file_path)
        # df = constant_interval(df)
        df['Value'] = smooth(df['Value'],10)
        data.append(df)
    return pd.concat(data)


def shadow_lineplot(Data, linestyle, marker, markersize, markevery, linewidth,
                    color, labels, save_path):
    fontsize = 16
    plt.figure(figsize=(8, 5))
    plt.rcParams['font.family'] = 'Times New Roman'
    for index, data in enumerate(Data):
        sns.lineplot(x='Step',
                     y='Value',
                     color=color[index],
                     data=data,
                     linewidth=linewidth[index],
                    #  marker=marker[index],
                    #  markevery=markevery[index],
                     linestyle=linestyle[index],
                    #  markersize=markersize[index],
                     alpha=1,
                     errorbar=("ci", 50),
                    #  err_kws={'alpha':0.2},
                     label=labels[index])
    plt.xlim(left=None, right=1e5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Training Step ', fontname='Times New Roman', fontsize=fontsize)
    plt.ylabel('Mean Reward of Episode', fontname='Times New Roman', fontsize=fontsize)
    plt.legend(fontsize=fontsize,title=None,loc='best')
    # plt.savefig(save_path)
    plt.show()


def multi_barplot(data, save_path):
    plt.figure(figsize=(10, 5))
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.barplot(
        data=data,
        x='Lane_counts',
        # x='Hdv_intervals',
        y='crash_count',
        hue='Model_types',
        palette='Set2')
    plt.xlabel('Lane_counts', fontname='Times New Roman', fontsize=14)
    plt.ylabel('crash_count', fontname='Times New Roman', fontsize=14)
    plt.savefig(save_path)
    plt.show()

def constant_interval(df, step_size=500):

    # 创建bins，步长为step_size
    bins = range(0, df['Step'].max() + step_size, step_size)

    # 使用cut函数将Step列划分为各个区间
    df['StepInterval'] = pd.cut(df['Step'], bins, right=False)

    # 对Value列进行聚合
    result = df.groupby('StepInterval')['Value'].mean().reset_index()

    # 将区间起点作为Step
    result['Step'] = result['StepInterval'].apply(lambda x: x.left)
    result = result.drop(columns='StepInterval')

    return result

if __name__ == "__main__":

    # lineplot
    color = ['indianred', 'teal', 'lightsalmon']
    color = ['#EFB336','#36AB60','indianred']
    # color = ['#614099','#1663A9','teal']

    marker = ['o', '^', 's']
    markevery = [5, 15, 15]
    linewidth = [2, 2, 2]
    linestyle = ['-', '--', ':']
    markersize = [8, 8, 8]

    # data_1 = get_line_data('data/MaskablePPO/Kinematic')
    # data_2 = get_line_data('data/A2C/Kinematic')

    # data_1 = get_line_data(r'data\DDQN\0')
    # data_2 = get_line_data(r'data\DDQN\2001')
    # data_3 = get_line_data(r'data\DDQN\2023')

    data_1 = get_line_data('./data/Graph_MaskablePPO')

    # labels = ['Kinematic', 'OccupancyGrid', 'Graph']
    # labels = ['MaskablePPO','A2C', 'DDQN']
    labels = ['SmartPL']


    data = [data_1]

    shadow_lineplot(data,
                    linestyle,
                    marker,
                    markersize,
                    markevery,
                    linewidth,
                    color,
                    labels=labels,
                    save_path='results/Extractor_MaskablePPO_Comparison.pdf',
                    )

    # barplot
    # data = pd.read_csv('./data/baselines.csv')
    # multi_barplot(data, save_path='./results/baselines_1.pdf')
    # print('over')
