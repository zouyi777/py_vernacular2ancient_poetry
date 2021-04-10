from py_vernacular2ancient_poetry.net_model import net_model
# from py_vernacular2ancient_poetry.net_model import net_model_bak as net_model
from py_vernacular2ancient_poetry.utils import pre_data_utils
import numpy as np
from py_common_dict import gen_com_dict
import os
import heapq

# 读取古诗白话文对
source_texts, target_texts = pre_data_utils.read_anct_poe_verna()
# 获取目标数据字典
tgt_vocab_size,tgt_dict,tgt_dict_reverse = pre_data_utils.gen_anct_poetry_dict()
# 生成解码器的输入和输出
target_in_texts,target_out_texts = pre_data_utils.gen_targt_in_out(target_texts,tgt_dict)

INUPT_LENGTH = max([len(i) for i in source_texts])
OUTPUT_LENGTH = max([len(i) for i in target_in_texts])

# 获取源数据字典
src_vocab_size,src_dict,_ = gen_com_dict.gen_dict2()

# 随机搜索
def sample(preds, temperature=1.0):
    '''
    当temperature=1.0时，模型输出正常
    当temperature小于1时时，模型输出比较保守,选择概率较大的值输出
    当temperature大于1时，模型输出比较open,选择概率较小的输出
    在训练的过程中可以看到temperature不同，结果也不同
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # print('preds：',preds)
    probas = np.random.multinomial(1, preds, 1) # 甩骰子
    # print('probas：', probas)
    return np.argmax(probas)
# 预测（贪心搜索）
def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    enc_outputs, enc_state_h, enc_state_c = encoder_inference.predict(source)
    state = [enc_state_h, enc_state_c]
    #第一个字符'\t',为起始标志
    predict_seq = np.array([[0]])
    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([enc_outputs, predict_seq]+state)
        # yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        # char_index = sample(yhat[0,-1,:])
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.array([[char_index]])
        if char_index == 1:#预测到了终止符则停下来
            break
        char = tgt_dict_reverse[str(char_index)]
        output += char
    return output

# 集束预测古诗
def predict_beam_poetry(source,encoder_inference, decoder_inference, n_steps, k=3):
    #先通过推理encoder获得预测输入序列的隐状态
    enc_outputs, enc_state_h, enc_state_c = encoder_inference.predict(source)
    state = [enc_state_h, enc_state_c]
    #第一个字符'\t',为起始标志
    predict_seq = np.array([[tgt_dict["sos"]]])
    output = []
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps -1):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([enc_outputs, predict_seq]+state)
        # yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        # char_index = sample(yhat[0,-1,:])
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        list_prob = yhat[0][0].tolist()
        probs_k = heapq.nlargest(k,list_prob)
        prob_indexs = [list_prob.index(prob_k) for prob_k in probs_k]
        # predict_seq = np.array([[char_index]])
        # if char_index == 1:#预测到了终止符则停下来
        #     break
        for prob_index in prob_indexs:
            if prob_index == tgt_dict["eos"]:
                continue
            poetry = tgt_dict_reverse[prob_index]
            output.append(poetry)
    return output

# 预测均衡输出(集束搜索)
def infer_beam_search(source,encoder_inference, decoder_inference, n_steps, k=3):
    # 先通过推理encoder获得预测输入序列的隐状态
    enc_outputs, enc_state_h, enc_state_c = encoder_inference.predict(source)
    enc_state_outputs = [enc_state_h, enc_state_c]

    predict_seq = [[0]]
    states_curr = {0: enc_state_outputs}
    seq_scores = [[predict_seq, 1.0, 0]]
    resList = []
    for _ in range(n_steps): # 时间步长
        cands = list()
        states_prev = states_curr
        for i in range(len(seq_scores)): # 候选词，长度等于束宽，即K值
            seq, score, state_id = seq_scores[i]
            dec_inputs = np.array(seq[-1:]) # 取最后一列
            dec_states_inputs = states_prev[state_id]
            # 解码器decoder预测输出
            dense_outputs, dec_state_h, dec_state_c = decoder_inference.predict([enc_outputs, dec_inputs] + dec_states_inputs)
            prob = dense_outputs[0][0]
            states_curr[i] = [dec_state_h, dec_state_c]

            k_prob = heapq.nlargest(k,prob)
            list_prob = prob.tolist()
            for item_prob in k_prob:
                cand = [seq + [[list_prob.index(item_prob)]], score * item_prob, i]
                cands.append(cand)
        seq_scores = heapq.nlargest(k, cands, lambda d: d[1])
    for i in range(len(seq_scores)):
    # for i in range(1):
        res = []
        for item in seq_scores[i][0]:
            if item[0] !=0 and item[0] != 1:
                res.append(tgt_dict_reverse[str(item[0])])
        resList.append(res)
    return resList
    # return resList[0]

def predict_ancient(texts):
    # model_path = "model/ancient_poetry_weights_bak.h5"
    model_path = "model/ancient_poetry_weights.h5"
    realpath = os.path.realpath(model_path)
    model_train = None
    if model_train == None:
        model_train = net_model.Seq2Seq(src_vocab_size + 1, tgt_vocab_size)
        model_train.load_weights(realpath)
    encoder_infer =None
    if encoder_infer == None:
        encoder_infer = net_model.encoder_infer(model_train)
    decoder_infer = None
    if decoder_infer == None:
        decoder_infer = net_model.decoder_infer(model_train,encoder_infer)
        # decoder_infer = net_model.decoder_infer(model_train)
    encoder_input = pre_data_utils.gen_sequence_without_onehot([texts], src_dict, INUPT_LENGTH)
    # out = predict_chinese(encoder_input, encoder_infer, decoder_infer, OUTPUT_LENGTH, tgt_vocab_size)
    out = predict_beam_poetry(encoder_input, encoder_infer, decoder_infer, OUTPUT_LENGTH,k=2)
    # out = infer_beam_search(encoder_input, encoder_infer, decoder_infer, OUTPUT_LENGTH)
    return out;

if __name__ == '__main__':
    texts = "过春风十里"
    out = predict_ancient(texts)
    print(texts)
    print(out)

