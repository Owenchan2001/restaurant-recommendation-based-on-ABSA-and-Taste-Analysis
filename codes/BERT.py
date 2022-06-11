import os
import time
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm #用于生成进度条
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn import functional as F

#模型任务：对中文商家评论进行情感极性二分类

class ClassifyModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_labels, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path) #加载BERT预训练模型
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, num_labels) #线性分类器，参数为输入输出的第二维度大小
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

    def save_model(self):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        # Save a trained model, configuration and tokenizer
        model_to_save = ( self )

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = "outputs/bert-chinese"
        output_config_file = "outputs/bert_config.json"

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

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
        labels = dataset.iloc[:, -1].tolist()
        labels = [i+2 for i in labels]
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
        t_labels = torch.tensor(labels, dtype=torch.long)

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
    raw_train_data = pd.read_csv(filepath[0],usecols=[1,17])
    raw_test_data = pd.read_csv(filepath[1],usecols=[1,17])
    print(raw_train_data)
    class_counter=[0,0,0,0]
    for index,row in raw_train_data.iterrows():
        class_counter[row['dish_taste']+2]+=1
    max_class=max(class_counter)
    class_counter=[1 for item in class_counter]
    print(class_counter)
    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForSingleSentence(bert_tokenizer=bert_tokenizer)
    # 产生输入句数据和迭代器
    train_data = processor.get_input(raw_train_data, max_seq_len)
    test_data = processor.get_input(raw_test_data, max_seq_len)
    weights = torch.FloatTensor(class_counter)
    sampler = WeightedRandomSampler(weights, int(len(train_data)/8), replacement=True)
    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    # 训练和测试的迷你批次数
    total_train_batch = math.ceil(len(raw_train_data) / batch_size)
    total_test_batch = math.ceil(len(raw_test_data) / batch_size)
    return train_iter, test_iter, total_train_batch, total_test_batch


def evaluate_accuracy(data_iter, net, device, batch_count):
    # 记录分类标签的准确性
    prediction_labels, true_labels = [], []
    with torch.no_grad():
        for count,batch_data in tqdm(enumerate(data_iter), desc='eval', total=batch_count):
            if count % 8 != 0: continue
            batch_data = tuple(t.to(device) for t in batch_data)
            # 获取给定的输出和模型给的输出
            labels = batch_data[-1]
            output = net(*batch_data[:-1])
            predictions = output.softmax(dim=1).argmax(dim=1)
            prediction_labels.append(predictions.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())

    return classification_report(np.concatenate(true_labels), np.concatenate(prediction_labels)) #打印训练报告

#多分类损失函数focal_loss
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=4, size_average=False):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

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


if __name__ == '__main__':
    batch_size, max_seq_len = 10, 150
    train_iter, test_iter, train_batch_count, test_batch_count = load_data(('./data/train/train.csv','./data/valid/valid.csv'), 'bert-base-chinese', max_seq_len, batch_size)    # 加载模型
    #model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=4)
    model = ClassifyModel('bert-base-chinese', num_labels=4, is_lock=True)
    print(model)

    optimizer = BertAdam(model.parameters(), lr=5e-05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = focal_loss() #损失函数是交叉熵损失

    for epoch in range(4):
        #牢记神经网络的训练过程为：在每个训练周期内，训练一次模型<数据批次化->神经网络前向计算返回结果->计算损失函数->损失函数反向传播->优化器步进->优化器梯度清零>->模型固定->无梯度条件下模型测试(模型输入测试数据->计算输出结果)
        start = time.time()
        model.train()
        # loss和精确度
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for step, batch_data in tqdm(enumerate(train_iter), desc='train epoch:{}/{}'.format(epoch + 1, 4)
                                    , total=train_batch_count):
            if step% 5 !=0: continue
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data

            logits = model(batch_seqs, batch_seq_masks, batch_seq_segments)
            loss = loss_func(logits, batch_labels)
            loss.backward()
            train_loss_sum += loss.item()
            logits = logits.softmax(dim=1)
            train_acc_sum += (logits.argmax(dim=1) == batch_labels).sum().item()
            n += batch_labels.shape[0]
            optimizer.step()
            optimizer.zero_grad()
        # 每一代都判断
        model.eval()
        result = evaluate_accuracy(test_iter, model, device,test_batch_count)
        print('epoch %d, loss %.4f, train acc %.3f, time: %.3f' %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, (time.time() - start)))
        print(result)

        # 模型验证
        s_i, b_c = getTargetComment("./sample1.csv", 'bert-base-chinese', max_seq_len, batch_size)
        prediction_labels = []
        with torch.no_grad():
            for b_d in tqdm(s_i, desc='eval', total=b_c):
                b_d = tuple(t.to(device) for t in b_d)
                # 获取给定的输出和模型给的输出
                output = model(*b_d[:-1])
                print(output)
                predictions = output.softmax(dim=1).argmax(dim=1)-2
                prediction_labels.append(predictions.detach().cpu().numpy())
        print(prediction_labels)
        s_i, b_c = getTargetComment("./sample2.csv", 'bert-base-chinese', max_seq_len, batch_size)
        prediction_labels = []
        with torch.no_grad():
            for b_d in tqdm(s_i, desc='eval', total=b_c):
                b_d = tuple(t.to(device) for t in b_d)
                # 获取给定的输出和模型给的输出
                output = model(*b_d[:-1])
                predictions = output.softmax(dim=1).argmax(dim=1)-2
                prediction_labels.append(predictions.detach().cpu().numpy())
        print(prediction_labels)

    model.save_model()