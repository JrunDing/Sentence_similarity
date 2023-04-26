"""
@Author: Jrun Ding
@Date: 2023.3.1
@Brief: calculate sentence similarity
@Coding: utf-8
"""


import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore") # 取消所有警告


'''
# example:
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go."
]
model = SentenceTransformer('bert-base-nli-mean-tokens') # 'bert-base-cased'  bert-base-nli-mean-tokens
sentence_embeddings_bert = model.encode(sentences) # ndarray [[]]
print(sentence_embeddings_bert[1])
print(cosine_similarity(
    [sentence_embeddings_bert[0]],
    sentence_embeddings_bert[1:]
))
'''


def cal_2sentences_similarity(sent1, sent2, pret_model='all-mpnet-base-v2'):
    """
    @brief：计算两个句子相似度   sentence-Bert SBERT/BERT bert-base-cased
    @param：['']  ['']  预训练模型，默认为SBERT，可改
        sbertdoc：https://www.sbert.net/docs/pretrained_models.html
        bert-base-cased  bert-base-uncased    all-MiniLM-L6-v2
    @return：sentence_similarity
    """
    sents = []
    sents.append(sent1[0])
    sents.append(sent2[0])
    model = SentenceTransformer(pret_model) # 使用 Windows 模型保存的路径在 C:\Users\[用户名]\.cache\torch\transformers\ 目录下，根据模型的不同下载的东西也不相同
    sentence_embeddings = model.encode(sents)
    sentence_similarity = (np.around(cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:]).tolist()[0],
                                     decimals=5))[0]
    return sentence_similarity


def cal_2texts_similarity(text1, text2, pret_model='all-MiniLM-L12-v2'):
    """
    @brief：计算两个文本相似度   sentence-Bert SBERT/BERT bert-base-cased
    @param：['','','']  ['','','']  预训练模型，默认为SBERT，可改cased BERT-base
    @return：文本相似度
    """
    sents_similarity = []
    for each in range(len(text1)):
        sents_similarity.append(cal_2sentences_similarity(text1[each:each+1], text2[each:each+1], pret_model))
    return sum(sents_similarity)/len(sents_similarity)


if __name__ == "__main__":

    a = ['There is a reason that roses have thorns.']
    b = ['Roses have thorns for a reason.']
    s = cal_2sentences_similarity(a, b)
    print("similarity:  ", s)

