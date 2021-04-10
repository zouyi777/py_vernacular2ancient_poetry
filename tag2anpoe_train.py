from tensorflow import keras
from py_vernacular2ancient_poetry.utils import pre_data_utils
from py_vernacular2ancient_poetry.net_model import NetModel
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

# 读取古诗白话文对
anctpoe_list, vernalabel_list = pre_data_utils.read_anct_poe_verna_label()
print("anctpoe_list=",len(anctpoe_list))
print("vernalabel_list=",len(vernalabel_list))
# 获取标签
train_y, class_len = pre_data_utils.gen_y(anctpoe_list)
# print("train_y", train_y)

# 获取白话词典
verna_dict, _, words_len = pre_data_utils.gen_verna_dict(vernalabel_list)

train_x = pre_data_utils.gen_x(vernalabel_list, verna_dict, pre_data_utils.input_length)
# print("train_x", train_x)


model = NetModel.LSTMAttentionNet.build(words_len+1, pre_data_utils.embedding_size, pre_data_utils.input_length, class_len)
model.summary()
optimizer = keras.optimizers.Adam(0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 自定义回调类，当loss小于某个值时，学习率设为0，不再更新,避免越过最优值
class LossHistory(Callback):  # 继承自Callback类
    def on_epoch_end(self, batch, logs={}):
        print("\r")
        loss = logs.get('loss')
        if loss < 5e-04:
            K.set_value(model.optimizer.lr, 1e-05)
        lr = K.get_value(model.optimizer.lr)
        print("     lr={}     loss={}".format(lr, loss))
loss_history = LossHistory()

# x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
model.fit(train_x, train_y, epochs=pre_data_utils.EPOCH, batch_size=pre_data_utils.BATCH_SIZE,
          validation_split=0.01, callbacks=[loss_history], shuffle=True)  # validation_split参数很重要，会影响孙旭验证准确率
# model.evaluate(x_test, y_test)
model.save('model/tag2anpoe_lstm.h5')