import numpy as np
import pandas as pd
from itertools import combinations
from tensorflow import keras

NUM_SAMPLES = 21
input_length = 10
embedding_size = 150
EPOCH = 100
BATCH_SIZE = 512

# 数据读取
def read_data():
    data_path = 'data/classic_sentence.txt'
    df = pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,:,]
    # df.columns=['inputs','targets','comments']
    df.columns=['inputs','targets']
    df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')
    src_texts = df.inputs.values.tolist()
    target_texts = df.targets.values.tolist()
    return src_texts,target_texts
# 读取数据，并切分原数据和目标数据
def read_anct_poe_verna():
    data_path = 'data/anct_poe_verna.txt'
    source_list = []
    target_list = []
    with open(data_path, "r", encoding='utf-8', ) as f:
        for line in f:
            content = line.replace('\n', '')  # 去掉换行符
            # content = content.replace('，', '')
            # content = content.replace('。', '')
            # content = content.replace('！', '')
            content = content.replace(' ', '')
            # content = content.replace('？', '')
            target, source = content.split("||")
            source_list.append(source)
            target_list.append(target)
    return source_list, target_list

def read_anct_poe_verna_label():
    data_path = 'data/anctpoe_famous_label.txt'
    anctpoe_list = []
    vernalabel_list = []
    with open(data_path, "r", encoding='utf-8', ) as f:
        for line in f:
            content = line.replace('\n', '')  # 去掉换行符
            content = content.replace(' ', '')
            anctpoe, vernalabel = content.split("||")
            vernalabel = vernalabel.split("、")

            for i in range(len(vernalabel)):
                zuhe = list(combinations(vernalabel, i + 1))
                for item in zuhe:
                    anctpoe_list.append(anctpoe)
                    vernalabel_list.append(list(item))

    return anctpoe_list, vernalabel_list

def gen_anct_poetry_dict():
    data_path = 'data/anct_poetry.txt'
    content_list = ["sos", "eos"]  # 预置开始结束符号
    with open(data_path, "r", encoding='utf-8', ) as f:
        for line in f:
            content = line.replace('\n', '')  # 去掉换行符
            content_list.append(content)
    content_list = sorted(list(set(content_list)))
    char_len = len(content_list)
    dict = {char: index for index, char in enumerate(content_list)}
    dict_reverse = {index: char for index, char in enumerate(content_list)}
    return char_len, dict, dict_reverse

# 生成词典
def gen_dict(texts):
    characters = sorted(list(set(texts)))
    char_len = len(characters)
    dict = {char: index+1 for index, char in enumerate(characters)}
    dict_reverse = {index+1: char for index, char in enumerate(characters)}
    return char_len, dict, dict_reverse

def gen_dict1(texts):
    characters = sorted(list(set(texts)))
    char_len = len(characters)
    dict = {char: index for index, char in enumerate(characters)}
    dict_reverse = {index: char for index, char in enumerate(characters)}
    return char_len, dict, dict_reverse

# 生成句向量,经过onehhot编码，返回数据是3维的
# 参数：sentence_list：句子列表;input_dict:词典
def gen_sequence(sentence_list,dict,sentence_len,vocab_size):
    sequence =[]
    for _,seq in enumerate(sentence_list):
        sentence_vector = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sentence_vector[char_index, dict[char]] = 1
        sequence.append(sentence_vector.tolist())
    return np.array(sequence)

# 生成句向量,不经过onehhot编码，返回数据是2维的
# 参数：sentence_list：句子列表;input_dict:词典
def gen_sequence_without_onehot(sentence_list,dict,sentence_len):
    sequence = []
    for _,seq in enumerate(sentence_list):
        sentence_seq = [0] * sentence_len
        for char_index, char in enumerate(seq):
            sentence_seq[char_index] = dict[char]
        sequence.append(sentence_seq)
    np_sequence = np.array(sequence)
    return np_sequence

# 生成目标句向量,经过onehhot编码，返回数据是3维的
def gen_sequence_targt(sentence_list,dict,sentence_len,vocab_size):
    sequence_in =[]
    sequence_out =[]
    for _,seq in enumerate(sentence_list):
        sen_vec_in = np.zeros((sentence_len, vocab_size))
        sen_vec_out = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sen_vec_in[char_index, dict[char]] = 1.0
            if char_index > 0:
                sen_vec_out[char_index - 1, dict[char]] = 1.0
        sequence_in.append(sen_vec_in.tolist())
        sequence_out.append(sen_vec_out.tolist())
    return np.array(sequence_in),np.array(sequence_out)

# 生成目标句向量,经过onehhot编码，返回数据是3维的，因为字典下标从1开始的，所以角标要减1，不要数组要越界
def gen_sequence_targt1(sentence_list,dict,sentence_len,vocab_size):
    sequence_out =[]
    for _,seq in enumerate(sentence_list):
        sen_vec_out = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            if char_index > 0:
                sen_vec_out[char_index - 1, dict[char] - 1] = 1.0
        sequence_out.append(sen_vec_out.tolist())
    return np.array(sequence_out)

def gen_sequence_targt2(sentence_list,sentence_len,vocab_size):
    sequence_out =[]
    for _,seq in enumerate(sentence_list):
        sen_vec_out = np.zeros((sentence_len, vocab_size))
        for char_index, char in enumerate(seq):
            sen_vec_out[char_index, char] = 1.0
        sequence_out.append(sen_vec_out.tolist())
    return np.array(sequence_out)

def gen_targt_in_out(target_texts,tgt_dict):
    target_in_texts = []
    target_out_texts = []
    for target_text in target_texts:
        target_in = [tgt_dict["sos"]]
        target_in.append(tgt_dict[target_text])
        target_in_texts.append(target_in)
        target_out = [tgt_dict[target_text]]
        target_out.append(tgt_dict["eos"])
        target_out_texts.append(target_out)
    return target_in_texts,target_out_texts

def gen_y(anpoe_list):
    char_len, dict, dict_reverse = gen_dict1(anpoe_list)
    train_y = []
    for char in anpoe_list:
        char_index = dict[char]
        sen_vec_out = np.zeros(char_len, dtype='int')
        sen_vec_out[char_index] = 1
        train_y.append(sen_vec_out.tolist())
    return np.array(train_y), char_len

# 生成白话文句子向量
def gen_x(verna_list, dict, sentence_len):
    train_x = []
    for _, seq in enumerate(verna_list):
        sentence_seq = [0] * sentence_len
        for char_index, char in enumerate(seq):
            sentence_seq[char_index] = dict[char]
            sentence_seq = sorted(sentence_seq, reverse=True)  #从大到小排序，规避标签顺序带来LSTM网络的影响
        train_x.append(sentence_seq)
    return np.array(train_x)

def gen_vern_label_x(verna_list,sentence_len):
    train_x = []
    for _, seq in enumerate(verna_list):
        sentence_seq = [0] * sentence_len
        for char_index, char in enumerate(seq):
            sentence_seq[char_index] = dict[char]
        train_x.append(sentence_seq)
    return np.array(train_x)

#  生成白话文词典
def gen_verna_dict(cuted_lines):
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(cuted_lines)
    verna_dict = tokenizer.word_index
    words_len = len(tokenizer.word_index)
    verna_dict_rev = tokenizer.index_word
    return verna_dict, verna_dict_rev, words_len

# 测试
if __name__ == '__main__':
    # texts = '人生入戏'
    # texts1 = '相见恨晚啊啊'
    # char_len,dict,dict_reverse = gen_dict(texts + texts1)
    # print("char_len:",char_len)
    # print("dict:",dict)
    # print("dict_reverse:",dict_reverse)
    # texts_list = [texts,texts1]
    # src_sequence = gen_sequence_without_onehot(texts_list,dict,7)
    # print("src_sequence:",src_sequence)
    # sequence,sequence_out = gen_sequence_targt(texts_list,dict,5,char_len)
    # print("sequence:",sequence)
    # print("sequence_out:",sequence_out)
    # source_list,target_list = read_ancient_poetry()
    # print("source_list",source_list)
    # print("target_list",target_list)
    dict_size,dict,dict_reverse = gen_anct_poetry_dict1()
    print(dict_size)
    print(dict)
    print(dict_reverse)