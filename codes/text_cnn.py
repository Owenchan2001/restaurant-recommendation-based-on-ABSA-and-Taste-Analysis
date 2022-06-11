import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors

from data_2_tensor import *
from tqdm import tqdm
from sklearn.metrics import classification_report


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size=300, num_of_header=3, hidden_droput=0.3):
        super(SelfAttention, self).__init__()
        if hidden_size % num_of_header != 0: raise ValueError()
        self.num_of_header = num_of_header
        self.attention_head_size = int(hidden_size/num_of_header)
        self.all_head_size = hidden_size

        self.Q = nn.Linear(input_size, self.all_head_size)
        self.K = nn.Linear(input_size, self.all_head_size)
        self.V = nn.Linear(input_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(hidden_droput)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_of_header, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_Q_layer = self.Q(input_tensor)
        mixed_K_layer = self.K(input_tensor)
        mixed_V_layer = self.V(input_tensor)

        Q_layer = self.transpose_for_scores(mixed_Q_layer)
        K_layer = self.transpose_for_scores(mixed_K_layer)
        V_layer = self.transpose_for_scores(mixed_V_layer)

        attention_scores = torch.matmul(Q_layer, K_layer.transpose(-1,-2))
        attention_scores /= math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        Context_layer = torch.matmul(attention_probs, V_layer)
        Context_layer = Context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = Context_layer.size()[:-2] + (self.all_head_size,)
        Context_layer = Context_layer.view(*new_context_layer_shape)
        return Context_layer

def text_2_tensor(label, max_len= 200, file_path='./trainer.txt'):
    vocab = dict()
    vocab['None'] = 0
    embedding_model = word2vec.Word2Vec.load('./trainer.model')
    vocab_list = [k for k,_ in embedding_model.wv.key_to_index.items()]
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        vocab[word] = i + 1
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        pointer = 0
        while True:
            pointer += 1
            line = f.readline()
            cent = [0 for i in range(max_len)]
            if len(line) == 0: break
            if pointer % 5 != 0: continue
            sentence = line.split()
            for index,word in enumerate(sentence):
                if index >= max_len: break
                if word not in vocab.keys():
                    vocab[word] = len(vocab)
                    cent[index] = vocab[word]
                else:
                    cent[index] = vocab[word]
            data.append(cent)
    length=len(label)
    data = torch.tensor(data, dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)
    print(len(data), len(label))
    dataset = TensorDataset(data, label)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True), math.floor(length / batch_size), len(vocab)


class TextCnn(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout=0.5):
        super(TextCnn, self).__init__()
        # 词嵌入模型预训练
        embedding_model = word2vec.Word2Vec.load('./trainer.model')
        word2idx = {'None': 0}
        vocab_list = [(k, embedding_model.wv[k]) for k, v in embedding_model.wv.key_to_index.items()]
        embeddings_matrix = np.zeros((len(embedding_model.wv.key_to_index.items()) + 1, embedding_model.vector_size))
        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            word2idx[word] = i + 1
            embeddings_matrix[i + 1] = vocab_list[i][1]
        embeddings_matrix = torch.tensor(embeddings_matrix, dtype=torch.float)
        # 初始化keras中的Embedding层权重
        self.embedding_table = nn.Embedding(len(embeddings_matrix), embed_dim, _weight= embeddings_matrix)
        self.embedding_table.requires_grad_ = False
        # 注意力机制
        self.attn = SelfAttention(embed_dim, hidden_size=embed_dim, num_of_header=2, hidden_droput=0.3)
        # text_cnn
        Ci = 1
        Co = kernel_num
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.embedding_table(x.long())
        x = self.attn(x)
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        return logit

# 测试分类结果的准确性
def accuracy(data_iter, model, batch_count):
    prediction_labels, true_labels = [], []
    with torch.no_grad():
        for count, batch_datas in tqdm(enumerate(data_iter), desc='eval', total=batch_count):
            features, targets = batch_datas
            output = model(features)
            pred = output.softmax(dim=1).argmax(dim=1)
            prediction_labels.append(pred.detach().numpy())
            true_labels.append(targets.detach().numpy())
    return classification_report(np.concatenate(true_labels), np.concatenate(prediction_labels))

#在回归评分时使用
def valid_loss(data_iter,model,batch_count):
    loss_func = nn.MSELoss()
    with torch.no_grad():
        for count, batch_datas in tqdm(enumerate(data_iter), desc='eval', total=batch_count):
            print('________________________________{}__________________________________'.format(count))
            features, targets = batch_datas
            output = model(features)
            loss = loss_func(output, targets)
            print(targets, output, loss)

if __name__ == '__main__':
    # 所需参数
    attr = 17
    batch_size = 10
    embed_dim, class_num, kernel_num, kernel_sizes = 300, 3, 2, [2, 3, 4, 5]
    model = TextCnn(embed_dim, class_num, kernel_num, kernel_sizes)
    # 获取训练数据
    train_labels = Get_Label(attr)
    train_data_loader, train_batch, vocab_len = text_2_tensor(train_labels)
    # 获取测试数据
    test_labels = Get_Label(attr, file_path='./data/valid/valid.csv')
    test_data_loader, test_batch, _ = text_2_tensor(test_labels, file_path='./valid.txt')
    # 模型训练与测试
    loss_func = nn.CrossEntropyLoss(torch.tensor([9, 1.1, 0.95], dtype=torch.float))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-04)
    for epoch in range(100):
        start = time.time()
        model.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for step, batch_data in tqdm(enumerate(train_data_loader), desc='train epoch:{}/{}'.format(epoch + 1, 100), total=train_batch):
            feature, target = batch_data
            optimizer.zero_grad()
            logit = model(feature)
            loss = loss_func(logit,target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            logits = logit.softmax(dim=1)
            train_acc_sum += (logits.argmax(dim=1) == target).sum().item()
            n += target.shape[0]
        model.eval()
        result = accuracy(test_data_loader, model, test_batch)
        print('epoch %d, loss %.4f, train acc %.3f, time: %.3f' %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, (time.time() - start)))
        print(result)
