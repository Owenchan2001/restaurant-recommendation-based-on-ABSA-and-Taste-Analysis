import math
import random
import numpy as np
import pandas as pd
import torch
from gensim.models import word2vec
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import TensorDataset


def get_attr_name(attr):
    # 根据索引获取属性名
    a = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
         'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
         'price_level', 'price_cost_effective', 'price_discount',
         'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
         'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
         'others_overall_experience', 'others_willing_to_consume_again'][attr - 2]
    return a

def Get_Label(attr, file_path='./data/train/train.csv'):
    # 将csv文件转化为[属性分类列表]
    label = []
    a = get_attr_name(attr)
    raw_data = pd.read_csv(file_path, usecols=[attr])
    safer, pointer = 5, 0  # 内存守护者
    for index, row in raw_data.iterrows():
        if row[a] == -2: continue
        pointer += 1
        if pointer % safer != 0: continue
        label.append(row[a]+1)
    return label

def Get_Mainstream(mainstream=50, token_path='./trainer.txt', frequency_path='./frequency.txt'):
    # 提取分词文本中若干个词频最高的关键词
    with open(frequency_path,'r',encoding='utf-8') as k:
        # 获取词频字典
        frequecy = eval(k.read())
    result = []
    safer,pointer = 5, 0 #内存守护者
    with open(token_path,'r',encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) == 0: break
            pointer += 1
            if pointer % safer != 0: continue
            # 获取mainstream个关键词(补0操作在word2vec时实现)
            sentence = line.split()
            length = len(sentence)
            if length <= mainstream:
                result.append(sentence)
            else:
                remainder = {}
                for index in range(length):
                    remainder[index] = frequecy[sentence[index]]
                remainder = sorted(list(remainder.items()),key=lambda t: t[1],reverse=True)
                if len(remainder) >= mainstream:
                    remainder = remainder[:mainstream]
                else:
                    # 随机生成补充的索引位置
                    make_up = random.sample(range(0,len(sentence)), mainstream-len(remainder))
                    for x in make_up:
                        remainder.append((x,0))
                remainder = sorted(remainder, key=lambda t: t[0])
                single_result = []
                for indice, _ in remainder:
                    single_result.append(sentence[indice])
                result.append(single_result)
    return result

def Get_DataLoader(mainstream_result, label, batch_size, num_worker=0, mainstream=50, model_path='./trainer.model', dtype=torch.long, sampler=False):
    # 利用word2vec生成数据并构造数据迭代器和批次数
    length = len(label)
    class_counter = [0, 0, 0]
    for k in range(length):
        class_counter[label[k]] += 1
    label = torch.tensor(label, dtype=dtype)
    vectors = []
    model = word2vec.Word2Vec.load(model_path)
    for sentence in mainstream_result:
        make_up = mainstream - len(sentence)
        single_vector = []
        for word in sentence:
            vec = model.wv[word].tolist()
            single_vector.append(vec)
        if make_up > 0:
            # 不足50个关键词的进行补0
            zeros = [0 for j in range(300)]
            for i in range(make_up):
                single_vector.append(zeros)
        vectors.append(single_vector)
    vectors = torch.tensor(vectors, dtype=torch.float)
    print(len(vectors),len(label))
    dataset = TensorDataset(vectors, label)
    # 采样器
    if sampler:
        print(class_counter)
        class_counter = [1/math.sqrt(j) for j in class_counter]
        print(class_counter)
        weights = torch.FloatTensor(class_counter)
        sample = WeightedRandomSampler(weights, len(label), replacement=True)
        return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sample, num_workers=num_worker,
                          drop_last=True), math.floor(length / batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True), math.floor(length / batch_size)

if __name__ == '__main__':
    attr = 17
    labels = Get_Label(attr)
    sentences = Get_Mainstream()
    data_loader, batch_size  = Get_DataLoader(sentences, labels, 10, num_worker=4)
    for step, batch_data in enumerate(data_loader):
        if step > 10: break
        feature, target= batch_data
        print(feature.shape)
        print(target.shape)
