from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as kl
from tensorflow.keras import backend as K
from tensorflow.keras import models

#  自定义Self_Attention注意力机制
class SelfAttention(kl.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        # 为该层创建一个可训练的权重 inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.output_dim), initializer='uniform', trainable=True)
        super(SelfAttention, self).build(input_shape)  # 一定要在最后调用它
    def call(self, x):
        vecQ = K.dot(x, self.kernel[0])
        vecK = K.dot(x, self.kernel[1])
        vecV = K.dot(x, self.kernel[2])
        # print("vecQ.shape", vecQ.shape)
        # print("K.permute_dimensions(vecK, [0, 2, 1]).shape", K.permute_dimensions(vecK, [0, 2, 1]).shape)

        QK = K.batch_dot(vecQ, K.permute_dimensions(vecK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        # print("QK.shape", QK.shape)

        V = K.batch_dot(QK, vecV)
        return V

    # 必须重写get_config方法，不然保存模型会报错.
    # 将__init___函数中定义的变量保存一下
    def get_config(self):
        config = {"output_dim": self.output_dim}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

# LSTM+注意力机制循环神经网络
class LSTMAttentionNet:
    @staticmethod
    def build(src_vocab_size, embedding_size, input_len, class_len):
        inputs = kl.Input(shape=(input_len,), name="encode_input")
        x = kl.Embedding(src_vocab_size, embedding_size, input_length=input_len)(inputs)
        x = SelfAttention(64)(x)
        x = kl.LSTM(64)(x)

        output = kl.Dense(class_len, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=output)
        return model

class CNNConv1DNet:
    @staticmethod
    def build(src_vocab_size, embedding_size, input_len, classes):
        model = Sequential()
        model.add(kl.Embedding(src_vocab_size, embedding_size, input_length=input_len))
        # model.add(SelfAttention(64))  # 注意力机制
        model.add(kl.Conv1D(64, 2, padding="same", activation="relu"))
        model.add(kl.MaxPooling1D(pool_size=2))
        model.add(kl.Dropout(0.5))

        model.add(kl.Conv1D(64, 2, padding="same", activation="relu"))
        model.add(kl.MaxPooling1D(pool_size=2))
        model.add(kl.Dropout(0.5))

        model.add(kl.Flatten())
        model.add(kl.Dense(128, activation='relu'))
        model.add(kl.Dense(128, activation='relu'))

        model.add(kl.Dense(classes, activation='softmax'))
        return model

if __name__ == '__main__':
    model = LSTMAttentionNet.build(100, 100, 3, 10)
    model.summary()