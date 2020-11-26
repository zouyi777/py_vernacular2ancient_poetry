import numpy as np
import pandas as pd

NUM_SAMPLES = 21

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
def read_ancient_poetry():
    data_path = 'data/ancient_poetry.txt'
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
            target,source=content.split("||")
            source_list.append(source)
            target = '\t'+ target+'\n'
            target_list.append(target)
    return source_list,target_list
# 生成词典
def gen_dict(texts):
    characters = sorted(list(set(texts)))
    char_len = len(characters)
    dict = {char: index+1 for index, char in enumerate(characters)}
    dict_reverse = {index+1: char for index, char in enumerate(characters)}
    return char_len,dict,dict_reverse

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

# 测试
if __name__ == '__main__':
    texts = '\t人生入戏'
    texts1 = '\t相见恨晚啊啊'
    char_len,dict,dict_reverse = gen_dict(texts + texts1)
    print("char_len:",char_len)
    print("dict:",dict)
    print("dict_reverse:",dict_reverse)
    texts_list = [texts,texts1]
    src_sequence = gen_sequence_without_onehot(texts_list,dict,7)
    print("src_sequence:",src_sequence)
    # sequence,sequence_out = gen_sequence_targt(texts_list,dict,5,char_len)
    # print("sequence:",sequence)
    # print("sequence_out:",sequence_out)
    # source_list,target_list = read_ancient_poetry()
    # print("source_list",source_list)
    # print("target_list",target_list)