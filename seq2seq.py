from keras_seq2seq_english2chinese.net_model import net_model
from keras_seq2seq_english2chinese.utils import pre_data_utils
from common_dict import gen_com_dict
from tensorflow import keras

BATCH_SIZE = 256
EPOCH = 500

input_texts, target_texts= pre_data_utils.read_ancient_poetry()

INUPT_LENGTH = max([len(i) for i in input_texts])
OUTPUT_LENGTH = max([len(i) for i in target_texts])
sentence_len = max(INUPT_LENGTH,OUTPUT_LENGTH)

# 获取字典
vocab_size,dict,_ = gen_com_dict.gen_dict1()

# 句子向量化
encoder_input = pre_data_utils.gen_sequence_without_onehot(input_texts,dict,INUPT_LENGTH)
decoder_input = pre_data_utils.gen_sequence_without_onehot(target_texts,dict,OUTPUT_LENGTH)
decoder_output = pre_data_utils.gen_sequence_targt1(target_texts,dict,OUTPUT_LENGTH,vocab_size)


model_train = net_model.Seq2Seq(vocab_size)
# optimizer = keras.optimizers.RMSprop(0.01)
optimizer = keras.optimizers.Adam(0.01)
model_train.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
model_train.summary()
# model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.1, shuffle=True)
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH, shuffle=True)
model_train.save_weights("model/ancient_poetry_weights.h5")

