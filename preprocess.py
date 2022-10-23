

import opensmile as of
import utils.opts as opts
'''
提取特征
'''

if __name__ == '__main__':

    config = opts.parse_opt()

    of.get_data(config, config.data_path, config.train_feature_path, train=True)
