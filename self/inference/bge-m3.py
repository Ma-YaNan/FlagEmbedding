import concurrent
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import time

import threading
import torch
import multiprocessing

class ModelCache:
    _instance_lock = threading.Lock()
    _predict_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with cls._instance_lock:
                if not hasattr(cls, '_instance'):
                    cls._instance = super(ModelCache, cls).__new__(cls)
                    cls._instance._model = BGEM3FlagModel('BAAI/bge-m3',devices='cuda', use_fp16=True)
        return cls._instance

    @property
    def model(self):
        return self._model
    
    # @model.setter
    # def model(self, new_model):
    #     self._model = new_model

    def predict(self, text):
        with self._predict_lock:
            output = self.model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
            dense, sparse, multiv = output['dense_vecs'], output['lexical_weights'], output['colbert_vecs']
            sparse = self.model.convert_id_to_token(sparse)
            return dense, sparse, multiv

    


text = ['稀疏检索中的每个分词的得分是独立计算的，得分通常反映了该分词与文档或查询的相关性。这些得分并没有经过标准化处理，也不会自然地加起来等于 1。',
        '在一些特定的模型中，可能会对得分进行归一化，但这取决于具体的实现。例如，在 概率模型 或某些 神经网络 中，可能会使用 softmax 等归一化技术使得得分归一化到 [0, 1] 范围内，但即使在这种情况下，分词的得分和也不一定等于 1。',
        '某些基于概率的模型或归一化技术可能会将得分映射到 0 到 1 之间，但这并不意味着它们的总和会等于 1。即使在进行归一化时，得分和也通常表示的是每个分词在查询-文档匹配中的相对重要性，而不是其概率或比例。',
        '查询和文档之间的相似性：查询中的某个词可能和文档中的多个位置匹配，因此在计算分数时，会考虑到查询中分词和文档中分词的重合情况。',
        'TF（Term Frequency）：一个词在文档中出现的次数。IDF（Inverse Document Frequency）：词在所有文档中出现的频率，用于反映一个词的稀有度，常常用于减小常见词的影响。文档长度：文档的长度可能对得分产生影响，尤其是在 TF-IDF 和 BM25 中，通常会对文档长度进行归一化。',
        '如果你有一个基于深度学习的模型（如 BERT 等），可以考虑通过模型来学习不同分词对词项得分的影响，而不是使用传统的求和或平均方法。在这种情况下，模型可能会根据语境自动学习每个分词的重要性，并为每个词项计算一个加权的得分。']


def example1():
    model = BGEM3FlagModel('BAAI/bge-m3',devices='cuda', use_fp16=True)
    print(model)
    sentences_1 = ["What is BGE M3?", "Defination of BM25",'File /home/ma/.vscode-server/extensions/ms-python.debugpy-2024.14.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py, line 71, in <module>cli.mai']

    output = model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    # output = model(text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    dense, sparse, multiv = output['dense_vecs'], output['lexical_weights'], output['colbert_vecs']
    sparse = model.convert_id_to_token(sparse)
    # r = sum(sparse[1].values())
    # print(r)
    print(sparse)
    return dense, sparse

def example2():
    print(ModelCache()._instance_lock)
    dense, sparse, multiv = ModelCache().predict(text)
    return dense, sparse

def example3():
    '''
    GPU
    多进程要设置为spawn模式，fork模式会报错
    worker数量设置为2，已达到最好效果，此时gpu利用率为100%

    CPU
    '''
    multiprocessing.set_start_method('spawn') 
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = [executor.submit(example2) for _ in range(5000)]
        for f in concurrent.futures.as_completed(results):
            r = f.result()
        pass


if __name__ == "__main__":
    # model = ModelCache()
    # s = time.time()
    # for i in range(10000):
    #     example2()
    # e = time.time()
    # print(e-s)
    # print(ModelCache()._instance_lock)
    # print(ModelCache()._instance_lock)
    s = time.time()
    # 1: 79.8  2: 55.6  3:55.7 4:56.8
    example3()
    e = time.time()
    print(e-s)

