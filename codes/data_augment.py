import csv
import random
import time

import pandas as pd
import numpy as np
import gensim
import math
import torch
import jieba
import torch.nn as nn
from gensim.models import doc2vec
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from sklearn.metrics import silhouette_score, classification_report
from Bio.Cluster import kcluster
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 设置停用词表
from tqdm import tqdm


def StopwordLoader(file_path='./data/stopwords.txt'):
    stopword = set(line.strip() for line in open(file_path, encoding='UTF-8').readlines())
    return stopword


# 数据读取
def get_attr_name(attr):
    # 根据索引获取属性名
    a = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
         'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
         'price_level', 'price_cost_effective', 'price_discount',
         'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
         'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
         'others_overall_experience', 'others_willing_to_consume_again'][attr - 2]
    return a


def DataLoader_1(attr, file_path='./data/train/train.csv'):
    # 将csv文件转化为[[分词列表],...]和[属性分类列表]
    sentences, label = [], []
    a = get_attr_name(attr)
    raw_data = pd.read_csv(file_path, usecols=[1, attr])
    for index, row in raw_data.iterrows():
        word_list = [k for k in jieba.cut(row['content'])]
        words = []
        for i in range(len(word_list)):
            if word_list[i] not in stop_word:
                words.append(word_list[i])
        label.append(row[a])
        sentences.append(words)
    return sentences, label


def DataLoader_2(sentences, mode='use', num_workers=10, context=10, neg=4,
                 num_features=300, min_word_count=1, downsampling=1e-3):
    # 将分词列表进行填充，并生成词嵌入向量
    # 训练或加载词向量模型
    if mode == 'train':
        documents = []
        TaggededDocument = gensim.models.doc2vec.TaggedDocument
        for p, sentence in enumerate(sentences):
            document = TaggededDocument(sentence, tags=[p])
            documents.append(document)
        model = doc2vec.Doc2Vec(documents, workers=num_workers,
                                vector_size=num_features, min_count=min_word_count,
                                window=context, negative=neg, sample=downsampling)
        model.init_sims(replace=True)
        # 保存模型，供日後使用
        model.save("trainer.model")
    else:
        model = doc2vec.Doc2Vec.load('./trainer')
    return model.infer_vector(['韭菜', '盒子', '贼', '几把', '好吃'], alpha=0.025)


def get_one_class_data(model, sentences, label, expected_class):
    # 加载某一类数据对应的词嵌入向量构成的列表
    vector_list = []
    length = len(label)
    for index in range(length):
        if label[index] == expected_class:
            vector_list.append(model.infer_vector(sentences[index], alpha=0.025))
    return vector_list


def get_prop(label):
    # 统计类别数目
    class_counter = [0, 0, 0, 0]
    for l in label:
        class_counter[l + 2] += 1
    return class_counter


# 数据预处理：数据聚类
def cos_similarity(x, y):
    # 余弦相似度
    sums = sum(x[i] * y[i] for i in range(len(x)))
    len_x = sum(i ** 2 for i in x)
    len_y = sum(i ** 2 for i in y)
    div = math.sqrt(len_x) * math.sqrt(len_y)
    return 1 - sums / div


def class_clustering(one_class_data):
    length = len(one_class_data)
    max_score = -1.01
    best_k = 2
    k_clusters = range(2, 11)
    my_metric = distance_metric(type_metric.USER_DEFINED, func=cos_similarity)
    for k in k_clusters:
        # 聚类部分
        clusterid, error, nfound = kcluster(one_class_data, k, dist='u', npass=100)
        # 利用轮廓系数评估部分
        silhouette_avg = silhouette_score(one_class_data, clusterid, metric='cosine')
        print(silhouette_avg)
        if silhouette_avg < 0.013: break
        if silhouette_avg > max_score:
            best_k = k
            max_score = silhouette_avg
    # 得到最终的聚类结果
    initial_centers = kmeans_plusplus_initializer(one_class_data, best_k).initialize()
    kmeans_instance = kmeans(one_class_data, initial_centers, metric=my_metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_centers = kmeans_instance.get_centers()
    mini_labels = [ 0 for p in range(length)]
    for kind in range(best_k):
        for index in clusters[kind]:
            mini_labels[index] = kind
    return best_k, mini_labels


def produce(one_class_data, best_k, mini_labels, expected_num):
    # 利用聚类数、聚类结果和期望数目完成一类数据的欠/过采样
    length = len(mini_labels)
    num_class = [0 for i in range(best_k)]
    data_class = [[] for i in range(best_k)]
    for i in range(length):
        num_class[mini_labels[i]] += 1
        data_class[mini_labels[i]].append(one_class_data[i])
    expected_change = expected_num - length
    if expected_change > 0:
        # 需要增加样本
        for i in range(best_k):
            increment = int(expected_change * (length - num_class[i]) / (length * (best_k - 1)))
            if increment <= num_class[i]:
                choice = random.sample(range(0, num_class[i]), increment)
            else: choice = [random.randint(0, num_class[i]) for l in range(increment)]
            for index in choice:
                one_class_data.append(data_class[num_class[i]][index])
    elif expected_change < 0:
        one_class_data = []
        # 需要减少样本
        for i in range(best_k):
            decrement = abs(int(expected_change * num_class[i] / length))
            choice = random.sample(range(0, num_class[i]), decrement)
            choice = sorted(choice)
            pointer = 0
            for index in choice:
                while pointer < index and pointer < num_class[i]:
                    one_class_data.append(data_class[num_class[i]][index])
                    i += 1
                i += 1
    return one_class_data


def in_class(labels):
    # 将-2~1的分类改成0~3
    length = len(labels)
    for index in range(length):
        labels[index] += 2
    return labels


def out_class(labels):
    # 将0~3的分类改成-1~2
    length = len(labels)
    for index in range(length):
        labels[index] -= 2
    return labels


# 模型：CNN
class TextCnn(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout=0.5):
        super(TextCnn, self).__init__()
        Ci = 1
        Co = kernel_num
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        return logit

# 生成迷你批次数据
def DataLoader_3(all_class_data, all_class_label,batch_size):
    all_class_data = torch.tensor(all_class_data, dtype=torch.float)
    all_class_label = torch.tensor(all_class_label, dtype=torch.long)
    training_data = TensorDataset(all_class_data,all_class_label)
    data_iter = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    total_train_batch = math.ceil(len(all_class_data) / batch_size)
    return data_iter, total_train_batch


# 测试分类结果的准确性
def accuracy(data_iter, model, batch_count):
    prediction_labels, true_labels = [], []
    with torch.no_grad():
        for count, batch_datas in tqdm(enumerate(data_iter), desc='eval', total=batch_count):
            features, targets = batch_datas
            output = model(features)
            pred = output.spftmax(dim=1).argmax(dim=1)
            prediction_labels.append(pred.detach().numpy())
            true_labels.append(targets.detach().numpy())
    return classification_report(np.concatenate(true_labels), np.concatenate(prediction_labels))


if __name__ == '__main__':
    # 准备常量
    stop_word = StopwordLoader()
    embed_dim, class_num, kernel_num, kernel_sizes = 300, 4, 2, [3, 4, 5]
    attr = 16
    batch = 10
    # 加载数据
    train_data, train_label = DataLoader_1(attr, './data/train/train.csv')
    train_total = len(train_label)
    test_data, test_label = DataLoader_1(attr, './data/valid/valid.csv')
    model = doc2vec.Doc2Vec.load('./trainer.model')
    '''# 进行均衡采样
    final_train_data, one_class_data, final_train_label = [], [], []
    for expected in range(-2,2):
        one_class_data = get_one_class_data(model, train_data, train_label, expected)
        best_k, mini_labels = class_clustering(one_class_data)
        one_class_data = produce(one_class_data, best_k, mini_labels, int(train_total/4))
        fake_kind = expected + 2
        for embedding in one_class_data:
            final_train_label.append(fake_kind)
            final_train_data.append(embedding)
        one_class_data = [] #节约内存'''
    # 随机打乱采样
    final_train_data, final_train_label, final_test_data, final_test_label = [], [], [], []
    for expected in range(-2, 2):
        one_class_data = get_one_class_data(model, train_data, train_label, expected)
        one_class_test = get_one_class_data(model, test_data, test_label, expected)
        fake_kind = expected + 2
        for embedding in one_class_data:
            final_train_label.append(fake_kind)
            final_train_data.append(embedding)
        for embedding in one_class_test:
            final_test_label.append(fake_kind)
            final_test_data.append(embedding)
        one_class_data = []
    random.seed(attr)
    random.shuffle(final_train_data)
    random.seed(attr)
    random.shuffle(final_train_label)
    # 进行训练
    model = TextCnn(embed_dim, class_num, kernel_num, kernel_sizes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-04)
    train_iter, batch_length = DataLoader_3(final_train_data, final_train_label, batch)
    test_iter, t_batch_length = DataLoader_3(final_test_data, final_test_label, batch)
    for epoch in range(4):
        start = time.time()
        model.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for step, batch_data in tqdm(enumerate(train_iter), desc='train epoch:{}/{}'.format(epoch + 1, 4)
                , total=batch_length):
            feature, target = batch_data
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            logits = logit.softmax(dim=1)
            train_acc_sum += (logits.argmax(dim=1) == target).sum().item()
            n += target.shape[0]
        model.eval()
        result = accuracy(test_iter, model, t_batch_length)
        print('epoch %d, loss %.4f, train acc %.3f, time: %.3f' %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, (time.time() - start)))
        print(result)