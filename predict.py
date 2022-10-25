import numpy as np
import opensmile as of
import utils.opts as opts
import soundfile as sf
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

    data_name = 'data'
    config = opts.parse_opt()
    audio_path = config.data_dir
    from keras import models
    import os
    from shutil import copyfile
    if not os.path.exists(config.data_dir + 'temp'):
        os.mkdir(config.data_dir + 'temp')
    if not os.path.exists(config.data_dir + 'temp/features/'):
        os.mkdir(config.data_dir + 'temp/features/')
    if not os.path.exists(config.data_dir + 'result'):
        os.mkdir(config.data_dir + 'result')
    predict_path = config.data_dir + 'temp/features/'
    if not os.path.exists(config.data_dir + 'temp/path_result'):
        os.mkdir(config.data_dir + 'temp/path_result')
        os.mkdir(config.data_dir + 'temp/label_result')
        os.mkdir(config.data_dir + 'temp/final_result')
    if not os.path.exists(config.data_dir + f'temp/features'):
        os.mkdir(config.data_dir + f'temp/features')
    predict_new_path = config.data_dir + f'temp/features/'
    model = models.load_model(os.path.join(config.checkpoint_path, config.checkpoint_name + '.h5'))
    for file in os.listdir(audio_path): # 重新创建
        if file in ['result','temp']:
            continue
        copyfile(os.getcwd() + '/features/single_feature.csv', config.data_dir + 'temp/features/single_feature.csv')
        if not os.path.exists(config.data_dir + f'temp/path_result/{file}'):
            os.mkdir(config.data_dir + f'temp/path_result/{file}')
        if not os.path.exists(config.data_dir + f'temp/label_result/{file}'):
            os.mkdir(config.data_dir + f'temp/label_result/{file}')
        if not os.path.exists(config.data_dir + f'temp/final_result/{file}'):
            os.mkdir(config.data_dir + f'temp/final_result/{file}')
        if not os.path.exists(config.data_dir + f'temp/features/{data_name}'):
            os.mkdir(config.data_dir + f'temp/features/{data_name}')
        if not os.path.exists(config.data_dir + f'temp/features/{data_name}/{file}'):
            os.mkdir(config.data_dir + f'temp/features/{data_name}/{file}')
        for files in os.listdir(os.path.join(audio_path,file)):
            result_path = config.data_dir + f'temp/path_result/{file}/predict_{files}_path.csv'
            result_path2 = config.data_dir + f'temp/label_result/{file}/predict_{files}_result.csv'
            result_path3 = config.data_dir + f'temp/final_result/{file}/predict_{files}_result.csv'
            if not os.path.exists(predict_new_path + f'{data_name}/{file}/{files}.csv'):
                of.get_new_data(config, audio_path + file + '/' + files,
                                predict_new_path + f'{data_name}/{file}/{files}.csv', result_path, train=False)
            test_feature = of.load_feature(config, predict_new_path + f'{data_name}/{file}/{files}.csv', train=False)

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
                    writer2.writerow([row["path"].split('/')[-2],strftime("%H:%M:%S", gmtime(current_second)),row["path"].split('/')[-1].split(".")[0].split("-")[-1],row["label"]])

    import pandas as pd
    import os
    import glob

    path = config.data_dir + 'temp/final_result'

    for file in os.listdir(config.data_dir + 'temp/final_result'):
        # for files in os.listdir(os.path.join(path,file)):
        #     print(files)
        all_csv = glob.glob(os.path.join(path, file) + r'\*.csv')
        all_data_frames = []
        for csv in all_csv:
            data_frame = pd.read_csv(csv, encoding='gbk')
            all_data_frames.append(data_frame)
        data_frame_concat = pd.concat(all_data_frames, axis=0, ignore_index=True)
        data_frame_concat.to_csv(config.data_dir + f'result/{file}.csv', index=False, encoding='gbk')
        print('合并完成!')