from gensim.models import word2vec

def read_and_train(file_path='./trainer.txt', model_path='./trainer.model'):
    # 从文本文件中读取分词结果，并提交给word2vec进行训练，并保存word2vec模型
    sentences = []
    with open(file_path,'r',encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) == 0: break
            sentence = line.split()
            sentences.append(sentence)
    model = word2vec.Word2Vec(sentences, workers=10, min_count=0, vector_size=300, alpha=0.025, window=4, negative=4)
    model.save(model_path)

def continue_train(file_path='./valid.txt', model_path='./trainer.model'):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) == 0: break
            sentence = line.split()
            sentences.append(sentence)
    model=word2vec.Word2Vec.load(model_path)
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_path)

if __name__=='__main__':
    #read_and_train()
    #continue_train()
    models = word2vec.Word2Vec.load('./trainer.model')
    print(models.wv['好吃'])
    print(models.wv['雌性'])
    print(models.wv['crasyones'])
    print(models.wv['缸内'])
