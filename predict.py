import numpy as np
from utils.common import Radar, play_audio
import opensmile as of
import utils.opts as opts
import soundfile as sf
import shutil
import csv
def reshape_input(data):
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data


'''
该文件是测试，打标签的
predict(): 预测音频情感

输入:
    audio_path: 要预测的音频路径
	model: 加载的模型

输出: 预测结果保存在result中，final_result即为最终输出
'''

if __name__ == '__main__':
    audio_path = 'dataset/0926'
    predict_path = 'features/'
    config = opts.parse_opt()
    from keras import models
    import os


    if not os.path.exists(f'result/path_result'):
        os.mkdir(f'result/path_result')
        os.mkdir(f'result/label_result')
        os.mkdir(f'result/final_result')
    if not os.path.exists(f'features/0925class'):
        os.mkdir(f'features/0925class')
    predict_new_path = 'features/0925class/'
    model = models.load_model(os.path.join(config.checkpoint_path, config.checkpoint_name + '.h5'))
    for file in os.listdir(audio_path): # 重新创建
        print(file)
        if not os.path.exists(f'result/path_result/{file}'):
            os.mkdir(f'result/path_result/{file}')
        if not os.path.exists(f'result/label_result/{file}'):
            os.mkdir(f'result/label_result/{file}')
        if not os.path.exists(f'result/final_result/{file}'):
            os.mkdir(f'result/final_result/{file}')
        if not os.path.exists(f'features/0925class/{file}'):
            os.mkdir(f'features/0925class/{file}')
        for files in os.listdir(os.path.join(audio_path,file)):
            print('课程',files)
            result_path = f'result/path_result/{file}/predict_{files}_path.csv'
            result_path2 = f'result/label_result/{file}/predict_{files}_result.csv'
            result_path3 = f'result/final_result/{file}/predict_{files}_result.csv'
            # if not os.path.exists(predict_new_path + f'{file}/{files}.csv'):
            #     of.get_new_data(config, os.path.join(os.path.join(audio_path, file), files),
            #                     predict_new_path + f'{file}/{files}.csv', result_path, train=False)
            test_feature = of.load_feature(config, predict_new_path + f'{file}/{files}.csv', train=False)

            test_feature = reshape_input(test_feature)

            result = model.predict(test_feature,batch_size = config.batch_size)

            result = np.argmax(result,axis = 1)
            print('Recogntion: ', [config.class_labels[int(result[i])] for i in range(len(result))])
            fields = ["label"] + [config.result_path]
            from time import strftime
            from time import gmtime


            with open(result_path,newline='') as infile, open(result_path2, "w",newline='') as outfile,open(result_path3, "w",newline='') as finalfile:
                r = csv.DictReader(infile)
                w = csv.DictWriter(outfile, fields, extrasaction="ignore")
                writer = csv.writer(outfile)
                writer.writerow(['label','path','time','current_time'])
                writer2 = csv.writer(finalfile)
                writer2.writerow(['课程名称', '时间', '视频编号', '情绪标签'])
                second = []
                for i,row in enumerate(r, start=1):
                    second.append(len(sf.read(row["path"])[0]) / sf.read(row["path"])[1])
                    if i ==1:
                        current_second = 0
                    else:
                        current_second += second[i-2]
                    row["label"] = config.class_labels[int(result[i-1])]
                    # print(row["label"],row["path"],second)
                    writer.writerow([row["label"],row["path"], second[i-1],current_second])
                    writer2.writerow([row["path"].split('\\')[-2],strftime("%H:%M:%S", gmtime(current_second)),row["path"].split('\\')[-1].split(".")[0].split("-")[-1],row["label"]])