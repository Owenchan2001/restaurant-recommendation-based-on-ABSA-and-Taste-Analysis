import re

import jieba
import pandas as pd

def StopwordLoader(file_path='./data/stopwords.txt'):
    stopword = set(line.strip() for line in open(file_path, encoding='UTF-8').readlines())
    return stopword

stop_word = StopwordLoader()

def get_attr_name(attr):
    # 根据索引获取属性名
    a = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
         'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
         'price_level', 'price_cost_effective', 'price_discount',
         'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
         'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
         'others_overall_experience', 'others_willing_to_consume_again'][attr - 2]
    return a

def Context_Tokenizer(attr, file_path='./data/train/train.csv', result_path='./trainer.txt'):
    # 将csv文件中的'context'转化为[[分词列表],...]
    sentences = []
    raw_data = pd.read_csv(file_path, usecols=[1, attr])
    a = get_attr_name(attr)
    for index, row in raw_data.iterrows():
        if row[a] == -2: continue
        content = row['content']
        c_len, poi = len(content), 0
        while poi<c_len:
            if content[poi] == '\n' or content[poi] == ' ' or content[poi] == '\t' or content[poi] == '\r':
                content = content[:poi] + content[poi+1:]
                c_len -= 1
            else: poi += 1
        word_list = [k for k in jieba.cut(content)]
        words = []
        for i in range(len(word_list)):
            if word_list[i] not in stop_word:
                words.append(word_list[i])
        sentences.append(words)
    with open(result_path,'w',encoding='utf-8') as f:
        for sentence in sentences:
            f.write(' '.join(sentence)+'\n')
    return sentences

def Context_Frequency(sentences, file_path='./frequency.txt'):
    # 统计分词结果中的词频，并写入文件
    word_dict = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else: word_dict[word] += 1
    # word_dict = sorted(list(word_dict.items()),key=lambda x: x[1],reverse=True) #按词频进行排序
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(str(word_dict))

if __name__=='__main__':
    attr = 17
    sentence1 = Context_Tokenizer(attr)
    sentence2 = Context_Tokenizer(attr, file_path='./data/valid/valid.csv', result_path='./valid.txt')
    for item in sentence2:
        sentence1.append(item)
    Context_Frequency(sentence1)
