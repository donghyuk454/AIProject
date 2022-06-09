#   *** Do not import any library except already imported libraries ***

import argparse
import logging
import os
from pathlib import Path
import json

import math
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hw4_util import AI_util
#   *** Do not import any library except already imported libraries ***

class Naive_Bayes(AI_util):
    def __init__(self, data_path):
        super(Naive_Bayes, self).__init__()

    def predict(self, test_data: list):
        """
        In this function, you have to predict the labels of test dataset.
        Also, save article ids and values of predicted labels.
        Do not use labels in test_data. It's not for the predict function.
        """
        ids = list()        # article ids
        preds = list()      # predicted labels
        values = list()     # log probabilities of predicted labels
        condVector = self.cond_prob
        ### EDIT FROM HERE ###
        labelList = ["entertainment", "finance", "lifestyle", "sports", "tv"]

        for docu in test_data:
            document = docu[1]
            id_ = docu[0]
            tfVector = self.get_tf_vector(document)
            vecLen = len(tfVector)
            tempValues = [float(0),float(0),float(0),float(0),float(0)] # f, l, t, s, e
            alpha = 0.5

            for i in range(0, 5):
                for j in range(0, vecLen):
                    tempValues[i] += (float(tfVector[j])) * math.log(condVector[i][0][j], 2)

            maxVal = tempValues[0]
            maxidx = 0

            for i in range(1, 5):
                if maxVal < tempValues[i] :
                    maxVal = tempValues[i]
                    maxidx = i

            ids.append(id_)
            preds.append(maxidx)
            values.append(round(maxVal, 2))
        
        ### EDIT UNTIL HERE ###

        return ids, values, preds
    
    def calc_conditional_prob(self, train_data: list):
        """
        In this function, you have to calculate conditional probabilities with training dataset.
        You can choose data structure of self.cond_prob as whatever you want to use.
        Then, you can use it in predict function.
        """
        self.cond_prob = None

        ### EDIT FROM HERE ###
        cond_list = list()
        tfVectorList = list()
        labelList = ["entertainment", "finance", "lifestyle", "sports", "tv"]
        labelTokenLen = [0, 0, 0, 0, 0]
        cpVectorList = list()
        tfVectorLen = 0
        alpha = 0.5

        # tf vector (id, tfvector, lable)
        for document in train_data:
            cnt = 0
            label = document[2]
            id_ = document[0]
            tfVector = self.get_tf_vector(document[1]) # get tf vector
            tfVectorLen = len(tfVector)
            tempList = list()

            for i in range(0, 5):
                if labelList[i] == label:
                    for tf in tfVector:
                        labelTokenLen[i] += float(tf)
                #labelTokenLen[i] += float(tfVectorLen)

            tempList.append(tfVector)
            tempList.append(label)
            tfVectorList.append(tempList)

        
        for label in labelList:
            tempLabel = list()
            tempLabel.append(label)
            temp = list()
            tempList = list()
            for j in range(0, tfVectorLen): 
                temp.append(float(0))
            tempList.append(temp)
            tempList.append(tempLabel)
            cpVectorList.append(tempList)

        # make conditinal prob
        for tfL in tfVectorList:
            tfVector = tfL[0]
            label_ = tfL[1]
            for i in range(0, 5):
                if label_ == labelList[i]:
                    for j in range(0, tfVectorLen):
                        cpVectorList[i][0][j] += float(tfVector[j])

        for i in range(0, 5):
            for j in range(0, tfVectorLen): 
                cpVectorList[i][0][j] += alpha
                cpVectorList[i][0][j] /= float(labelTokenLen[i])
        
        self.cond_prob = cpVectorList

        ### EDIT UNTIL HERE ###

    def get_tf_vector(self, document):
        """
        This function returns fixed size of TF vector for a document.
        You can use this function if you want to use.
        """
        tf_vec = np.zeros(len(self.word2idx), dtype=np.int32)
        
        for token in document:
            if token in self.word2idx.keys():
                tf_vec[self.word2idx[token]] += 1

        return tf_vec


def main(args, logger):
    data_path = os.path.join('./')

    nb_classifier = Naive_Bayes(data_path)
    logger.info("Classifier is initialized!")

    train_data = nb_classifier.load_data(data_path, 'train')
    logger.info("# of train data: {}".format(len(train_data)))
    test_data = nb_classifier.load_data(data_path, 'test')
    logger.info("# of test data: {}".format(len(test_data)))

    nb_classifier.calc_conditional_prob(train_data)
    ids, values, preds = nb_classifier.predict(test_data)
    labels = [nb_classifier.label2idx[y] for _, _, y in test_data]
    print(labels)
 
    accuracy = accuracy_score(labels, preds)
    precsion = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')

    precsion_ma = precision_score(labels, preds, average='macro')
    recall_ma = recall_score(labels, preds, average='macro')
    f1_ma = f1_score(labels, preds, average='macro')
    
    logger.info("Accuracy: {}%".format(accuracy * 100))
    logger.info("Precision: {}%".format(precsion * 100))
    logger.info("Recall: {}%".format(recall * 100))
    logger.info("F1: {}%".format(f1 * 100))

    with open(args.output_dir/'NB_.txt', 'w', encoding='utf-8') as f:
        f.write("Accuracy: {}%\n".format(str(accuracy * 100)))
        f.write("Precision: {}%\n".format(str(precsion * 100)))
        f.write("Recall: {}%\n".format(str(recall * 100)))
        f.write("F1: {}%\n\n".format(str(f1 * 100)))
        f.write("Precision_ma: {}%\n".format(str(precsion_ma * 100)))
        f.write("Recall_ma: {}%\n".format(str(recall_ma * 100)))
        f.write("F1_ma: {}%\n\n".format(str(f1_ma * 100)))
        
        ### EDIT FROM HERE ###
        """ Implement codes for the output text file. """
        length = len(ids)
        for i in range(0, length):
            if preds[i] == 0:
                f.write("{}\tfinance\t{}\n".format(str(ids[i]), str(values[i])))
        
        ### EDIT UNTIL HERE ###

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--std_name',
                        type=str,
                        default='이동혁',
                        help='Student name.')
    parser.add_argument('--std_id',
                        type=str,
                        default="2016311076",
                        help='Student ID')

    parser.add_argument('--selected_label',
                        type=str,
                        default='tv',
                        help="Label to write its scores down to output text file.")
    parser.add_argument('--output_dir',
                        type=Path,
                        default=Path('./'),
                        help="Path where output will be saved.")
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)

    main(args, logger)
