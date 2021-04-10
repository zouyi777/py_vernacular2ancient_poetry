from py_vernacular2ancient_poetry.net_model import net_model
from py_vernacular2ancient_poetry.utils import pre_data_utils
from py_common_dict import gen_com_dict
from tensorflow import keras
import numpy as np

BATCH_SIZE = 128
EPOCH = 100

# 读取古诗白话文对
source_texts, target_texts = pre_data_utils.read_anct_poe_verna()
# 获取古诗数据字典
tgt_vocab_size, tgt_dict, _ = pre_data_utils.gen_anct_poetry_dict()
# 生成解码器的输入和输出
target_in_texts, target_out_texts = pre_data_utils.gen_targt_in_out(target_texts, tgt_dict)

INUPT_LENGTH = max([len(i) for i in source_texts])
OUTPUT_LENGTH = max([len(i) for i in target_in_texts])

# 获取源数据字典
src_vocab_size, src_dict, _ = gen_com_dict.gen_dict2()


# 句子向量化
encoder_input = pre_data_utils.gen_sequence_without_onehot(source_texts,src_dict,INUPT_LENGTH)
decoder_input = np.array(target_in_texts)
decoder_output = pre_data_utils.gen_sequence_targt2(target_out_texts,OUTPUT_LENGTH,tgt_vocab_size)


model_train = net_model.Seq2Seq(src_vocab_size + 1, tgt_vocab_size)
# optimizer = keras.optimizers.RMSprop(0.01)
optimizer = keras.optimizers.Adam(0.01)
model_train.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
model_train.summary()
# model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.1, shuffle=True)
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH, shuffle=True)
model_train.save_weights("model/ancient_poetry_weights.h5")

