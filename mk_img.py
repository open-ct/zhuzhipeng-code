import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
该文件是画图的，根据统计得到的每节课的情绪标签按时间顺序画图，图片会保存在/result/img/
'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
audio_path = 'result/label_result'
for file in os.listdir(audio_path):  # 重新创建
    print(file)
    if not os.path.exists(f'result/img/{file}'):
        os.mkdir(f'result/img/{file}')
    for files in os.listdir(os.path.join(audio_path, file)):
        print(files)
        data = pd.read_csv(audio_path + f'/{file}/{files}', encoding="gbk")
        for i in ["angry-愤怒", "Delate-删除的", "disgust-厌恶蔑视", "fear-恐惧担忧", "happy-喜悦", "sad-悲伤", "surprised-惊讶",
                  "TS-师生互动"]:
            if data[data['label'] == i]['time'].sum() != 0:
                locals()[f'{i}_y'] = data[data['label'] == i]['time']
                locals()[f'{i}_z'] = data[data['label'] == i]['current_time']
            else:
                locals()[f'{i}_y'] = 0.1
                locals()[f'{i}_z'] = 0.1

        fig = plt.figure(figsize=(8, 4))  # 调整画布大小
        ax = plt.gca()
        ax.set_facecolor('oldlace')
        plt.grid(True)
        width = 0.5
        # p1 = plt.barh(['happy'],data['current_time'].tolist()[-1],width, color='white')
        label = ["angry-愤怒", "Delate-删除的", "disgust-厌恶蔑视", "fear-恐惧担忧", "happy-喜悦", "sad-悲伤", "surprised-惊讶", "TS-师生互动"]
        for i in label:
            locals()[f'p{label.index(i)}'] = plt.barh([i], locals()[f'{i}_y'], width, left=locals()[f'{i}_z'])
        plt.xlabel("时间（秒）")
        plt.title(f'{files}'.split('.')[0].split('predict_')[1])
        x = []
        plt.xticks()
        plt.xlim()
        plt.savefig(f'result/img/{file}_img/' + f'{files}'.split('.')[0].split('predict_')[1] + '.jpg')
