from tensorflow import keras
from tensorflow.keras.layers import Flatten, Input, Embedding, LSTM, Dense, Attention,Concatenate
from tensorflow.keras.models import Model

N_UNITS = 128
embedding_dim = 150

def create_model(n_input, n_output, n_units):
    # 训练阶段
    # encoder
    encoder_input = Input(shape=(None, n_input))
    # encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    # n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    _, encoder_h, encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h, encoder_c]
    # 保留下来encoder的末状态作为decoder的初始状态

    # decoder
    decoder_input = Input(shape=(None, n_output))
    # decoder的输入维度为中文字符数
    decoder = LSTM(n_units, return_sequences=True, return_state=True)
    # 训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)
    # 在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    # 输出序列经过全连接层得到结果

    # 生成的训练模型
    model = Model([encoder_input, decoder_input], decoder_output)
    # 第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出

    # 推理阶段，用于预测过程
    # 推断模型—encoder
    encoder_infer = Model(encoder_input, encoder_state)

    # 推断模型-decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]  # 上个时刻的状态h,c

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,
                                                                                 initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]  # 当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)  # 当前时刻的输出
    decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)

    return model, encoder_infer, decoder_infer

# 编码器，标准的RNN/LSTM模型，取最后时刻的隐藏层作为输出
class Encoder(keras.Model):
    def __init__(self, hidden_units, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)  # Embedding Layer
        self.encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="encode_lstm")# Encode LSTM Layer
    def call(self, inputs):
        encoder_embed = self.embedding(inputs)
        outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return outputs,state_h, state_c
# 解码器，有三部分输入，一是encoder部分的每个时刻输出，二是encoder的隐藏状态输出，三是decoder的目标输入
class Decoder(keras.Model):
    def __init__(self,hidden_units, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=False)  # Embedding Layer
        self.decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="decode_lstm") # Decode LSTM Layer
        self.attention = Attention()  # Attention Layer
        self.concatenate = Concatenate(axis=-1, name='concat_layer')  # Concatenate Layer
    def call(self,enc_outputs,dec_inputs, init_state):
        decoder_embed = self.embedding(dec_inputs)
        dec_outputs, state_h, state_c = self.decoder_lstm(decoder_embed, initial_state=init_state)
        attention_output = self.attention([dec_outputs, enc_outputs])
        concatenate_output = self.concatenate([dec_outputs, attention_output])  # 一定要把注意力输出和解码输出粘接起来
        return concatenate_output, state_h, state_c
# encoder和decoder模块合并，组成一个完整的seq2seq模型
def Seq2Seq(src_vocab_size,tgt_vocab_size):
    # Input Layer
    encoder_inputs = Input(shape=(None, ), name="encode_input")
    decoder_inputs = Input(shape=(None, ), name="decode_input")
    # Encoder Layer
    encoder = Encoder(N_UNITS, src_vocab_size, embedding_dim)
    enc_outputs,enc_state_h, enc_state_c = encoder(encoder_inputs)
    enc_states = [enc_state_h, enc_state_c]
    # Decoder Layer
    decoder = Decoder(N_UNITS, tgt_vocab_size, embedding_dim)
    dec_output, _, _ = decoder(enc_outputs,decoder_inputs, enc_states)
    # Dense Layer
    dense_outputs = Dense(tgt_vocab_size, activation='softmax', name="final_out_dense")(dec_output)
    # seq2seq model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
    return model
# 从seq2seq模型中获取Encoder子模块
def encoder_infer(model):
    encoder_my = model.get_layer('encoder')
    input_my = encoder_my.input
    output_my = encoder_my.output
    encoder_model = Model(inputs=input_my,outputs=output_my)
    return encoder_model
# 从seq2seq模型中获取Decoder子模块，这里没有直接从decoder层取，方便后续decoder的预测推断
def decoder_infer(model,encoder_model):
    encoder_output = encoder_model.get_layer('encoder').output[0]
    maxlen, hidden_units = encoder_output.shape[1:]

    dec_input = model.get_layer('decode_input').input
    enc_output = Input(shape=(maxlen, hidden_units), name='enc_output')
    input_state_h = Input(shape=(N_UNITS,))
    input_state_c = Input(shape=(N_UNITS,))
    dec_state_input = [input_state_h, input_state_c]  # 上个时刻的状态h,c

    decoder = model.get_layer('decoder')
    dec_outputs, out_state_h, out_state_c = decoder(enc_output, dec_input, dec_state_input)
    dec_states_out = [out_state_h, out_state_c]

    final_out_dense = model.get_layer('final_out_dense')
    dense_output = final_out_dense(dec_outputs)

    decoder_model = Model(inputs=[enc_output,dec_input]+dec_state_input,
                          outputs=[dense_output]+dec_states_out)
    return decoder_model

if __name__ == '__main__':
    model = Seq2Seq(10)
    model.summary()
    enc_model = encoder_infer(model)
    enc_model.summary()
    dec_model = decoder_infer(model)
    dec_model.summary()