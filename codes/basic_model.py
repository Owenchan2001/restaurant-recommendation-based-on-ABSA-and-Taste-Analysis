import time
import codecs
import csv
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm #用于生成进度条
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import TensorDataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from torch.autograd import Variable

#模型任务：对中文商家评论进行情感评分预测

#将分类标签处理为评分
def preOperation(file_path,save_path,weight=[2,1,1,3,4,2,4,4,4,3,4,2,2,4,4,4,2,4,8,8]):
    #将标签转化为评分
    raw_data=pd.read_csv(file_path)
    raw_data=raw_data.values
    f = open(save_path, 'w', encoding='utf-8-sig', newline="")
    c = csv.writer(f)
    c.writerow(["text" , "score"])
    for line in range(raw_data.shape[0]):
        text=raw_data[line,1]
        labels=raw_data[line,2:]
        sum_w = 0.0
        score = 0.0
        for i in range(20):
            if labels[i] == -2: continue
            if labels[i] == -1:
                score += weight[i]
                sum_w += weight[i]
            if labels[i] == 0:
                score += weight[i] * 3.5
                sum_w += weight[i]
            if labels[i] == 1:
                score += weight[i] * 5
                sum_w += weight[i]
        if sum_w==0: c.writerow([text , 3.5])
        else: c.writerow([text , score/sum_w])
    f.close()

#用于读取商家评论(生成一个数据迭代器)
def getTargetComment(file_path,pretrained_model_name_or_path, max_seq_len, batch_size):
    raw_comment_data = pd.read_csv(file_path)
    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForSingleSentence(bert_tokenizer=bert_tokenizer)
    # 产生输入句数据和迭代器
    comment_data = processor.get_input(raw_comment_data, max_seq_len)
    comment_iter = DataLoader(dataset=comment_data, batch_size=batch_size, shuffle=True)
    # 训练和测试的迷你批次数
    total_comment_batch = math.ceil(len(raw_comment_data) / batch_size)
    return comment_iter, total_comment_batch


class ClassifyModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path) #加载BERT预训练模型
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 1) #线性分类器，参数为输入输出的第二维度大小
        if is_lock:
            # 加载并冻结bert模型参数
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        #BertModel()说明：
        #参数
        #input_ids表示输入实例的张量，形状为(batch_size, sqe_len)
        #token_type_ids表示句子标记，如果其值为一个形状与input_ids相同的张量则表示一个实例包含两个句子
        #attention_mask表示传入的每个实例的长度，用于attention的mask
        #返回值
        #encoded_layer是长度为num_hidden_layers的形状为 (batch_size, sqe_len,hidden_size)的张量列表
        #pooled_output是形状为 (batch_size, hidden_size)的，代表了句子信息的张量，就是在BERT中最后一层encoder的第一个词[CLS]经过Linear层和激活函数Tanh后的张量
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled = self.dropout(pooled) #随机丢弃
        logits = self.classifier(pooled) #池化结果分类
        return logits

#对单句文本进行训练文本的转化
class DataProcessForSingleSentence(object):
    def __init__(self, bert_tokenizer, max_workers=10):
        """
        :param bert_tokenizer: 分词器
        :param max_workers: 最大并行数
        """
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset, max_seq_len=30):
        #加载数据集
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()
        #分词
        token_seq = list(self.pool.map(self.bert_tokenizer.tokenize, sentences))
        #获取定长序列及其mask(mask是指BERT一定概率把一些词变成完形填空的机制)
        #pool.map用于多进程并行处理，参数是函数名和该函数所需的参数，参数都是列表而且是一一对应的关系，这里返回的是每次函数返回三元组的列表
        result = list(self.pool.map(self.trunate_and_pad, token_seq,
                                    [max_seq_len] * len(token_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        #将数据转化为Long类型的张量
        t_seqs = torch.tensor(seqs, dtype=torch.long)
        t_seq_masks = torch.tensor(seq_masks, dtype=torch.long)
        t_seq_segments = torch.tensor(seq_segments, dtype=torch.long)
        t_labels = torch.tensor(labels, dtype=torch.float)

        return TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)

    def trunate_and_pad(self, seq, max_seq_len):
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0: (max_seq_len - 2)]
        # 添加分类标签
        seq = ['[CLS]'] + seq + ['[SEP]']
        # 将分词映射为ID
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        # 检查长度
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def load_data(filepath, pretrained_model_name_or_path, max_seq_len, batch_size):
    """
    加载excel文件，有train和test 的sheet
    :param filepath: 文件路径
    :param pretrained_model_name_or_path: 使用什么样的bert模型
    :param max_seq_len: bert最大尺寸，不能超过512
    :param batch_size: 小批量训练的数据
    :return: 返回训练和测试数据迭代器 DataLoader形式
    """
    raw_train_data = pd.read_csv(filepath[0])
    raw_test_data = pd.read_csv(filepath[1])
    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForSingleSentence(bert_tokenizer=bert_tokenizer)
    # 产生输入句数据和迭代器
    train_data = processor.get_input(raw_train_data, max_seq_len)
    test_data = processor.get_input(raw_test_data, max_seq_len)
    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    # 训练和测试的迷你批次数
    total_train_batch = math.ceil(len(raw_train_data) / batch_size)
    total_test_batch = math.ceil(len(raw_test_data) / batch_size)
    return train_iter, test_iter, total_train_batch, total_test_batch


def evaluate_accuracy(data_iter, net, device, batch_count):
    # 记录分类标签的准确性
    total_loss=0.0
    test_count=0
    with torch.no_grad():
        for count,batch_data in tqdm(enumerate(data_iter), desc='eval', total=batch_count):
            if count % 30 !=0: continue
            test_count+=1
            batch_data = tuple(t.to(device) for t in batch_data)
            # 获取给定的输出和模型给的输出
            predict = batch_data[-1]
            output = net(*batch_data[:-1])
            total_loss += (predict-output) ** 2

    return total_loss/test_count if test_count else 0

if __name__ == '__main__':
    batch_size, max_seq_len = 4, 200
    train_iter, test_iter, train_batch_count, test_batch_count = load_data(('./data/train/c_train.csv','./data/valid/c_valid.csv'), 'bert-base-chinese', max_seq_len, batch_size)
    # 加载模型
    #model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
    model = ClassifyModel('bert-base-chinese', is_lock=True)
    print(model)

    optimizer = BertAdam(model.parameters(), lr=5e-05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = nn.MSELoss() #回归问题的损失函数是均方误差损失

    for epoch in range(4):
        #牢记神经网络的训练过程为：在每个训练周期内，训练一次模型<数据批次化->神经网络前向计算返回结果->计算损失函数->损失函数反向传播->优化器步进->优化器梯度清零>->模型固定->无梯度条件下模型测试(模型输入测试数据->计算输出结果)
        start = time.time()
        model.train()
        # loss和精确度
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for step, batch_data in tqdm(enumerate(train_iter), desc='train epoch:{}/{}'.format(epoch + 1, 4)
                                    , total=train_batch_count):
            if step % 5 != 0: continue
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data

            logits = model(batch_seqs, batch_seq_masks, batch_seq_segments)
            loss = loss_func(logits, batch_labels)
            loss.backward()
            train_loss_sum += loss.item()
            n += batch_labels.shape[0]
            optimizer.step()
            optimizer.zero_grad()
        # 每一代都判断
        model.eval()
        # 模拟两个商家评论情景
        s_i, b_c = getTargetComment("./sample1.csv", 'bert-base-chinese', max_seq_len, batch_size)
        t_c = 0
        score = 0.0
        with torch.no_grad():
            for b_d in tqdm(s_i, desc='eval', total=b_c):
                b_d = tuple(t.to(device) for t in b_d)
                # 获取给定的输出和模型给的输出
                output = model(*b_d[:-1])
                print(output)
                score += torch.sum(output)
                t_c += output.size(0)
        print(score / t_c)

        s_i, b_c = getTargetComment("./sample2.csv", 'bert-base-chinese', max_seq_len, batch_size)
        t_c = 0
        score = 0.0
        with torch.no_grad():
            for b_d in tqdm(s_i, desc='eval', total=b_c):
                b_d = tuple(t.to(device) for t in b_d)
                # 获取给定的输出和模型给的输出
                output = model(*b_d[:-1])
                print(output)
                score += torch.sum(output)
                t_c += output.size(0)
        print(score / t_c)

        result = evaluate_accuracy(test_iter, model, device,test_batch_count)
        print('epoch %d, loss %.4f, time: %.3f' %
              (epoch + 1, train_loss_sum / n, (time.time() - start)))
        print(result)

    torch.save(model, 'fine_tuned_chinese_bert.bin') #保存模型