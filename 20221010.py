import pandas as pd
import os
import glob
path = 'result/final_result'


for file in os.listdir('result/final_result'):
    print(file)
    # for files in os.listdir(os.path.join(path,file)):
    #     print(files)
    all_csv = glob.glob(os.path.join(path,file) + r'\*.csv')
    print(all_csv)
    all_data_frames = []
    for csv in all_csv:
        print(csv)
        data_frame = pd.read_csv(csv,encoding='gbk')
        all_data_frames.append(data_frame)
    data_frame_concat = pd.concat(all_data_frames,axis=0,ignore_index=True)
    data_frame_concat.to_csv(f'{file}.csv',index=False,encoding='gbk')
    print('合并完成!')