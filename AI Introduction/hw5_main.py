#   *** Do not import any library except already imported libraries ***

import argparse
import logging
import os
from pathlib import Path
import json

import math
import random
import numpy as np
from collections import Counter
from tqdm import tqdm, trange

from hw5_utils import AI_util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#   *** Do not import any library except already imported libraries ***

class KNN(AI_util):
    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k
    
    def predict(self, train_data, test_data):
        preds = None

        ### EDIT HERE ###
        trainLabels = list()
        labels = ["entertainment", "finance", "lifestyle", "sports", "tv"]
        
        trainDocuments = list()
        testDocuments = list()

        for content in train_data:
            document = content[1]
            label = content[2]
            trainDocuments.append(document)
            _id = content[0]
            for i in range(0, 5):
                if labels[i] == label:
                    temp = [_id, i]
                    trainLabels.append(temp)
                    break
        for content in test_data:
            document = content[1]
            testDocuments.append(document)

        preds = list()
        dataLen = len(trainLabels)

        tf = self.get_tf(trainDocuments)
        #tfidf = self.get_normalized_tfidf(tf, self.get_idf(trainDocuments))
        #bow = self.get_bow(trainDocuments)

        _tf = self.get_tf(testDocuments)
        #_tfidf = self.get_normalized_tfidf(_tf, self.get_idf(testDocuments))
        #_bow = self.get_bow(testDocuments)
        
        for m in tqdm(range(len(test_data)), desc="predict...: "):
            #cosbowList = list()
            costfList = list()
            #costfidfList = list()

            #distbowList = list()
            #disttfList = list()
            #disttfidfList = list()

            for n in range(len(trainDocuments)):
                label = train_data[n][2]
                labelNum = 0

                for i in range(0, 5):
                    if labels[i] == label:
                        labelNum = i
                        break

                #cosbow = self.get_cosine_sim(_bow[m], bow[n])
                costf = self.get_cosine_sim(_tf[m], tf[n])
                #costfidf = self.get_cosine_sim(_tfidf[m], tfidf[n])

                #distbow = self.get_euclidean_dist(_bow[m], bow[n])
                #disttf = self.get_euclidean_dist(_tf[m], tf[n])
                #disttfidf = self.get_euclidean_dist(_tfidf[m], tfidf[n])

                #cosbowList.append([cosbow, labelNum])
                costfList.append([costf, labelNum])
                #costfidfList.append([costfidf, labelNum])

                #distbowList.append([distbow, labelNum])
                #disttfList.append([disttf, labelNum])
                #disttfidfList.append([disttfidf, labelNum])
            checkCnt = [0, 0, 0, 0, 0]

            #temp__ = sorted(disttfidfList, key = lambda a : (a[0]))
            temp__ = sorted(costfList, key = lambda a : (a[0]), reverse=True)

            for i in range(self.k):
                checkCnt[temp__[i][1]] += 1

            max_ = 0
            num = 0

            for i in range(0, 5):
                if max_ < checkCnt[i]:
                    max_ = checkCnt[i]
                    num = i
            preds.append(num)

        ### END ###

        return preds
    
    def get_bow(self, documents):
        bow = None

        ### EDIT HERE ###
        bow = list()

        for document in documents: 
            bow_temp = np.zeros(len(self.word2idx), dtype=np.int32)
            for token in document:
                if token in self.word2idx.keys():
                    bow_temp[self.word2idx[token]] = 1
            bow.append(bow_temp)
        ### END ###

        return bow

    def get_tf(self, documents):
        tf = None
        ### EDIT HERE ###
        tf = list()

        for document in documents:
            tf_temp = np.zeros(len(self.word2idx), dtype=np.int32)
            for token in document:
                if token in self.word2idx.keys():
                    tf_temp[self.word2idx[token]] += 1
            tf.append(tf_temp)
        ### END ###

        return tf

    def get_idf(self, documents):
        idf = None

        ### EDIT HERE ###
        idf = list()
        idf_ = np.zeros(len(self.word2idx), dtype=np.int32)
        
        doculen = float(len(documents))

        for document in documents:
            temp = np.zeros(len(self.word2idx), dtype=np.int32)
            for token in document:
                if token in self.word2idx.keys() and temp[self.word2idx[token]] == 0:
                    temp[self.word2idx[token]] = 1
                    idf_[self.word2idx[token]] += float(1)
        for i in range(len(idf_)):
            if idf_[i] != 0:
                idf.append(math.log(doculen/idf_[i], 2))
        ### END ###
        return idf

    def get_normalized_tfidf(self, tf, idf):
        tfidf = None

        ### EDIT HERE ###
        tfidf = list()

        for temp in tf:
            tem = list()
            for i in range(len(idf)):
                tem.append(float(temp[i]) * idf[i])
            length = math.sqrt(sum(i**2 for i in tem))
            for i in range(len(tem)):
                tem[i] /= length
            tfidf.append(tem)
        ### END ###

        return tfidf

    def get_euclidean_dist(self, vec_a, vec_b):
        dist = None

        ### EDIT HERE ###
        dist = float(0)
        dist = math.sqrt(sum((float(vec_a[i]-vec_b[i]))**2 for i in range(len(vec_a))))
        ### END ###

        return dist
    
    def get_cosine_sim(self, vec_a, vec_b):
        sim = None

        ### EDIT HERE ###
        sim = float(0)
        len_a = float(0)
        len_b = float(0)
        for i in range(len(vec_a)):
            a = vec_a[i]
            b = vec_b[i]
            sim += a * b
            len_a += a**2
            len_b += b**2

        len_a = math.sqrt(len_a)
        len_b = math.sqrt(len_b)

        sim /= (len_a * len_b)
        ### END ###

        return sim


def main(args, logger):

    k = args.k
    knn = KNN(k)
    train_data = knn.load_data(args.data_dir, 'train')
    test_data = knn.load_data(args.data_dir, 'test')

    labels = [knn.label2idx[l] for _, _, l in test_data]
    preds = knn.predict(train_data, test_data)
    acc = knn.calc_accuracy(labels, preds)
    logger.info("Accuracy: {:.2f}%".format(acc * 100))

    ### EDIT ###
    """ Implement codes writing output file """
    std_name = "이동혁"
    std_id = "2016311076"
    metric_ = "Cosine similarity"
    #metric_ = "Euclidean distance"
    #input_ = "bag-of-words"
    input_ = "TF"
    #input_ = "TF-IDF"
    with open("./"+"{0}_{1}.txt".format(std_name, std_id), 'a', encoding='utf-8') as f:
        f.write("Metric: {}\n".format(metric_))
        f.write("Input: {}\n".format(input_))
        f.write("Accuracy: {:.2f}%\n\n".format(round(acc * 100, 2)))
    
    accuracy = accuracy_score(labels, preds)
    precsion = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')

    ma_precsion = precision_score(labels, preds, average='macro')
    ma_recall = recall_score(labels, preds, average='macro')
    ma_f1 = f1_score(labels, preds, average='macro')
    
    print("Accuracy: {}%".format(accuracy * 100))
    print("Precision: {}%".format(precsion * 100))
    print("Recall: {}%".format(recall * 100))
    print("F1: {}%".format(f1 * 100))

    print("ma_Precision: {}%".format(ma_precsion * 100))
    print("ma_Recall: {}%".format(ma_recall * 100))
    print("ma_F1: {}%".format(ma_f1 * 100))

    with open('./KNN_.txt', 'w', encoding='utf-8') as f:
        f.write("Accuracy: {}%\n".format(str(accuracy * 100)))
        f.write("Precision-micro: {}%\n".format(str(precsion * 100)))
        f.write("Recall-micro: {}%\n".format(str(recall * 100)))
        f.write("F1-micro: {}%\n\n".format(str(f1 * 100)))
        f.write("Precision-macro: {}%\n".format(str(ma_precsion * 100)))
        f.write("Recall-macro: {}%\n".format(str(ma_recall * 100)))
        f.write("F1-macro: {}%\n\n".format(str(ma_f1 * 100)))
        

    ### END ###
    if std_name == None: raise ValueError
    if std_id == None: raise ValueError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='./',
                        help="Path where dataset exist.")
    parser.add_argument('--output_dir',
                        type=str,
                        default='./',
                        help="Path where output will be saved.")

    parser.add_argument('--k',
                        type=int,
                        default=11)
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)

    main(args, logger)
