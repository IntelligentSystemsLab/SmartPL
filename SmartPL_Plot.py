# plot figures in SmartPL paper
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


from utils import *

# ---------------setting-----------------#

# color = ['indianred', 'teal', 'lightsalmon']
# color = ['#EFB336','#36AB60','indianred']
# color = ['#614099','#1663A9','teal','#EFB336']
# color = ['teal', 'lightsalmon','#EFB336']
# color = ['#BB97D3','#BDE3ED','#FFB78B','#EC8282']
color = ['teal','#BB97D3','#FFB78B','#EC8282']
# color = ['teal', '#A4E33D'] #safe
# color = ['teal', '#1663A9','#EFB336'] #extractor
marker = ['o', '^', 's','*']
markevery = [5, 5, 5,5]
linewidth = [2, 2, 2,2]
linestyle = ['-', '--', ':','-.']
markersize = [12, 12, 12,12]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2.0
fontsize = 20
# ---------------setting-----------------#

# training process
def line_plot():
    data_1 = get_line_data('data/Graph_MaskablePPO/reward')
    # data_1 = get_line_data('data/Graph_MaskablePPO/loss')
    
    data_2 = get_line_data('data/CoOP')
    data_3 = get_line_data('data/Plexe')
    data_4 =get_line_data('data/NosiyNet-MADQN')

    labels = ['SmartPL', 'CoOP', 'Plexe','NosiyNet-MADQN']

    # data = [data_1]
    data = [data_1, data_2, data_3,data_4]

    fontsize = 20
    fig,ax = plt.subplots(figsize=(6, 3))
    plt.rcParams['font.family'] = 'Times New Roman'
    for index, data in enumerate(data):
        sns.lineplot(x='Step',
                        y='Value',
                        color=color[index],
                        data=data,
                        linewidth=linewidth[index],
                        marker=marker[index],
                        markevery=markevery[index],
                        linestyle=linestyle[index],
                        markersize=markersize[index],
                        alpha=1,
                        errorbar=("ci", 50),    
                    #  err_kws={'alpha':0.2},
                        label=labels[index])
    plt.xlim(left=None, right=1e5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Training Step ', fontname='Times New Roman', fontsize=fontsize,weight='bold')
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=1.5, zorder=1)
    plt.ylabel('Mean of Training Reward', fontname='Times New Roman', fontsize=fontsize,weight='bold')
    # plt.ylabel('Value Loss', fontname='Times New Roman', fontsize=fontsize,weight='bold')

    plt.legend('')
    # plt.legend(fontsize=fontsize,title=None,loc='upper center',ncol=4)

    plt.savefig('results/smartpl_lineplot_reward.pdf')
    # plt.savefig('results/smartpl_lineplot_loss.pdf')
    plt.show()

# Deprecated: robust perfomance in different conditions
def box_plot():

    data = []
    # for filename in os.listdir(data_dir):
    #     if filename.endswith('.csv'):  # 检查是否为 CSV 文件
    #         file_path = os.path.join(data_dir, filename)  # 构建完整文件路径
    #     df = pd.read_csv(file_path)
    #     data.append(df)
    for seed in [0,2001,2023]:
        file_path = 'data/baselines/seed_{0}/final/combined_data.csv'.format(seed)
        df = pd.read_csv(file_path)
        data.append(df)
    data = pd.concat(data)

    lane_count = 2
    data_filtered = data[(data['Lane_counts'] == lane_count)]
    plt.rcParams['font.family'] = 'Times New Roman'
    fig,ax = plt.subplots(figsize=(5,4))
    fontsize = 20
    # plt.figure(figsize=(8, 5))   
    bar = sns.boxplot(
data=data_filtered,
    # hue='Model_types',
    x='Model_types',
    y='mean_reward',
    # y='mean_sim_time',
    # fill=False,
    linewidth=2,
    palette=color)
    for patch in bar.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.8))  # 设置透明度为0.3
    ax.set_xlabel('')
    plt.ylabel('Mean Reward', fontname='Times New Roman', fontsize=fontsize,weight='bold')
    # plt.ylabel('Mean Simulation Time', fontname='Times New Roman', fontsize=fontsize)
    # plt.legend(fontsize=fontsize,title=None,loc='upper center',ncol=4)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    ax.grid(color='gray', linestyle='--', linewidth=1)
    plt.ylim(bottom=1100,top=1850)
    plt.savefig('results/SmartPL_boxplot_lane{}_reward.eps'.format(lane_count))    
    plt.show()

# robust perfomance in different conditions
def bar_plot():
    plt.figure(figsize=(5, 4))
    data = []
    for seed in [0, 2001, 2023]:
        file_path = 'data/baselines/seed_{0}/final/combined_data.csv'.format(seed)
        df = pd.read_csv(file_path)
        data.append(df)
    data = pd.concat(data)

    interval = 3

    data_filtered = data[(data['Hdv_intervals'] == interval)]
    bars = sns.barplot(x='Model_types', y='mean_reward', data=data_filtered, palette=color, alpha=0.8,edgecolor='black', errcolor='black', errwidth=2, capsize=0.1,linewidth=2, zorder=2)
    

    # 设置图表属性
    patterns = ['//', '--', '||', '\\\\']
    for bar, pattern in zip(bars.patches, patterns):
        bar.set_hatch(pattern)
    plt.ylim(bottom=1100,top=1850)
    bars.grid(True, which='both', color='gray', linestyle='--', linewidth=1.5, zorder=1)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    bars.set_xlabel('')
    # bars.set_xticks('')
    children = plt.gca().get_children()    
    plt.legend([children[0], children[2],children[4], children[6]], ['SmartPL', 'CoOP','Plexe', 'NoisyNet-MADQN'],fontsize=fontsize,loc='upper center',ncol=4 )
    # plt.legend( fontsize=fontsize, title=None, 
    #        loc='upper center',ncol=4)
    plt.ylabel('Mean Reward', fontname='Times New Roman', fontsize=fontsize,weight='bold')
    # plt.savefig('results/SmartPL_barplot_interval{}_reward.eps'.format(interval))   
    plt.show()
    
def extractor():
    # color = ['#614099','#EFB336','indianred']
    data_1 = get_line_data('data/Graph_MaskablePPO/reward')
    data_2 = get_line_data('data/Graph_MaskablePPO/MLP')
    data_3 = get_line_data('data/Graph_MaskablePPO/CNN')

    labels = ['GAT Spatial Extractor', 'MLP Spatial Extractor','CNN Spatial Extractor']


    data = [data_1, data_2,data_3]

    fontsize = 20
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    for index, data in enumerate(data):
        sns.lineplot(x='Step',
                        y='Value',
                        color=color[index],
                        data=data,
                        linewidth=linewidth[index],
                        marker=marker[index],
                        markevery=markevery[index],
                        linestyle=linestyle[index],
                        markersize=markersize[index],
                        alpha=1,
                        errorbar=("ci", 50),    
                    #  err_kws={'alpha':0.2},
                        label=labels[index])
    plt.xlim(left=None, right=1e5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Training Step ', fontname='Times New Roman', fontsize=fontsize,weight='bold')

    plt.ylabel('Mean of Training Reward', fontname='Times New Roman', fontsize=fontsize,weight='bold')


    plt.legend(fontsize=fontsize,title=None,loc='best')

    plt.savefig('results/smartpl_extractor_lineplot.pdf')

    plt.show()

def safety_training():
    # color = ['#614099','#36AB60','indianred']
    data_1 = get_line_data('data/Graph_MaskablePPO/reward')
    data_2 = get_line_data('data/safe monitor/training')

    labels = ['With Safety Monitor', 'Without Safety Monitor']

    # data = [data_1]
    data = [data_1, data_2]

    fontsize = 20
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    for index, data in enumerate(data):
        sns.lineplot(x='Step',
                        y='Value',
                        color=color[index],
                        data=data,
                        linewidth=linewidth[index],
                        marker=marker[index],
                        markevery=markevery[index],
                        linestyle=linestyle[index],
                        markersize=markersize[index],
                        alpha=1,
                        errorbar=("ci", 50),    
                    #  err_kws={'alpha':0.2},
                        label=labels[index])
    plt.xlim(left=None, right=1e5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Training Step ', fontname='Times New Roman', fontsize=fontsize,weight='bold')

    plt.ylabel('Mean of Training Reward', fontname='Times New Roman', fontsize=fontsize,weight='bold')    

    plt.legend(fontsize=fontsize,title=None,loc='best')

    plt.savefig('results/smartpl_safety_lineplot.pdf')
    plt.show()


# bar_plot()
# box_plot()
line_plot()
# safety_training()
# extractor()