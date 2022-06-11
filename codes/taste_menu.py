import joblib
import jieba
import re
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from tokenize_and_frequency import stop_word
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

stopword = stop_word()

def sep_single(name):
    name_depart = jieba.cut(name)
    out = ''
    for word in name_depart:
        if word not in stopword and not re.match(r'[^A-Z^a-z^0-9^\u4e00-\u9fa5]', word):
            out += word + ' '
    return out

def sep_menu(file_path='./data/menu.csv'):
    # 将菜单分割为分词序列
    df = pd.read_csv(file_path, encoding='utf-8', usecols=[0])
    dish_name = []
    for index,row in df.iterrows():
        name_depart = sep_single(row['name'])
        dish_name.append(name_depart)
    return dish_name

def count_frequency(corpus, in_file):
    # 获取词频向量
    if os.path.exists(in_file):
        cnt_vector = joblib.load(in_file)
        cnt_tf = cnt_vector.transform(corpus)
    else:
        cnt_vector = CountVectorizer()
        cnt_tf = cnt_vector.fit_transform(corpus)
        print('主题词袋:', len(cnt_vector.get_feature_names()))
        joblib.dump(cnt_vector, in_file)
    return cnt_tf

def LDA(in_model, model_in_data, n=2):
    # 训练LDA模型
    if os.path.exists(in_model):
        lda = joblib.load(in_model)
        res = lda.transform(model_in_data)
    else:
        lda = LatentDirichletAllocation(n_components=n,
                                        # max_iter=5,
                                        # learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        res = lda.fit_transform(model_in_data)
        joblib.dump(lda, in_model)
    return res

def accuracy(model, file_path='./data/menu.csv'):
    df = pd.read_csv(file_path, encoding='utf-8')
    true_labels, prediction_labels = [], []
    for index,row in df.iterrows():
        name_depart = sep_single(row['name'])
        true_labels.append(row['taste'])
        prediction = model.fit(name_depart)
        prediction = np.argmax(prediction)
        prediction_labels.append()
    return classification_report(np.concatenate(true_labels), np.concatenate(prediction_labels))

if __name__ == '__main__':
    n = len(['酸','甜','苦','辣','咸','鲜'])
    data_list = sep_menu()
    cv_file = "./CountVectorizer.pkl"
    cnt_data_list = count_frequency(data_list, cv_file)
    model_file = "./lda_model.pk"
    docres = LDA(model_file, cnt_data_list, n=n)
    # 文档所属每个类别的概率
    LDA_corpus = np.array(docres)
    print('类别所属概率:\n', LDA_corpus)

    test_length = 6
    pre_list = ['甜甜花酿鸡', '盐酥鸡', '韭菜盒子', '几把', '九品芝麻蛋糕', '苦瓜汤']
    for i in range(test_length):
        pre_list[i] = sep_single(pre_list[i])
    pre_cnt_data_list = count_frequency(pre_list, cv_file)
    pre_docres = LDA(model_file, pre_cnt_data_list, n=n)
    print('预测数据概率:\n', np.array(pre_docres))

