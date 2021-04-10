from tensorflow import keras
from py_vernacular2ancient_poetry.utils import pre_data_utils
from py_common_dict import gen_com_dict
import numpy as np
import heapq
from py_vernacular2ancient_poetry.net_model import NetModel

# 读取古诗白话文对
anctpoe_list, vernalabel_list = pre_data_utils.read_anct_poe_verna_label()
# 获取古诗词典
char_len, dict, dict_reverse = pre_data_utils.gen_dict1(anctpoe_list)

# 获取白话字典
verna_dict, _, words_len = pre_data_utils.gen_verna_dict(vernalabel_list)

test = [['人', '离别', '团聚', '春风', '医治', '生老病死']]
test = pre_data_utils.gen_x(test, verna_dict, pre_data_utils.input_length)

model = keras.models.load_model('model/tag2anpoe_lstm.h5', custom_objects={'SelfAttention': NetModel.SelfAttention})
predictions = model.predict(test)[0]
argmax = np.argmax(predictions)
print(dict_reverse[argmax])