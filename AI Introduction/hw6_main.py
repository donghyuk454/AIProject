# -*- coding: utf-8 -*-
import math
import numpy as np
from tqdm import tqdm
from hw6_util import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Preprocessing(AI_util):
    def Calculate_Binary(self, data: List[Tuple[str, List[str], int]])  -> List[Tuple[str, List[float], int]]:
        binary = list()
        documents = list()
        documentID = list()
        categorys = list()

        for i in range(len(data)):
            documents.append(data[i][1])
            documentID.append(data[i][0])
            categorys.append(data[i][2])

        for i in range(len(documents)): 
            temp = list()
            binary_temp = np.zeros(len(self.word2idx), dtype=np.int32)
            for token in documents[i]:
                if token in self.word2idx.keys():
                    binary_temp[self.word2idx[token]] = float(1)
            temp.append(documentID[i])
            temp.append(binary_temp)
            temp.append(categorys[i])
            binary.append(temp)
        return binary

    def Calculate_TF(self, data: List[Tuple[str, List[str], int]])  -> List[Tuple[str, List[float], int]]:
        tf = list()

        documents = list()
        documentID = list()
        categorys = list()

        for i in range(len(data)):
            documents.append(data[i][1])
            documentID.append(data[i][0])
            categorys.append(data[i][2])

        for i in range(len(documents)):
            temp = list()
            tf_temp = np.zeros(len(self.word2idx), dtype=np.int32)
            for token in documents[i]:
                if token in self.word2idx.keys():
                    tf_temp[self.word2idx[token]] += 1
            temp.append(documentID[i])
            temp.append(tf_temp)
            temp.append(categorys[i])
            tf.append(temp)


        return tf

    def Calculate_TF_IDF_Normalization(self, data: List[Tuple[str, List[str], int]], data_type: str)  -> List[Tuple[str, List[float], int]]:
        idf = list()
        idf_ = np.zeros(len(self.word2idx), dtype=np.int32)

        documents = list()
        documentID = list()
        categorys = list()

        for i in range(len(data)):
            documents.append(data[i][1])
            documentID.append(data[i][0])
            categorys.append(data[i][2])

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

        tfidf = list()
        tf_ = self.Calculate_TF(data)
        tf = list()

        for list_ in tf_:
            tf.append(list_[1])

        for i in range(len(tf)):
            #tem = list()
            tem = np.zeros(len(self.word2idx), dtype=np.float32)
            temp = list()
            for j in range(len(idf)):
                #tem.append(float(tf[i][j]) * idf[j])
                tem[j] = float(tf[i][j]) * idf[j]
            length = math.sqrt(sum(j**2 for j in tem))
            for j in range(len(tem)):
                tem[j] /= length
            temp.append(documentID[i])
            temp.append(tem)
            temp.append(categorys[i])
            tfidf.append(temp)

        return tfidf

class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
        ### EDIT HERE ###
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.error = float(0)
        self.befor_error = float(1)

        self.output_answer = np.ones(self.output_size, dtype=np.float32)

        self.array_input = np.ones(self.input_size, dtype=np.float32)
        self.array_hidden = np.ones(self.hidden_size, dtype=np.float32)
        self.array_output = np.ones(self.output_size, dtype=np.float32)

        self.w_input = 0.01*np.random.randn(self.input_size, self.hidden_size)
        self.w_output = 0.01*np.random.randn(self.hidden_size, self.output_size)
    
        self.temp_w_input = np.full((self.input_size, self.hidden_size),0.0)
        self.temp_w_output = np.full((self.hidden_size, self.output_size),0.0)
        
        ### END ###

    def forward(self, input):
        # This function is for forwarding.
        ### EDIT HERE ###
        self.array_input = input
        self.array_hidden = np.tanh(np.dot(self.array_input, self.w_input)) # self.array_input[i]*self.w_input[i][j]

        ex = np.dot(self.array_hidden, self.w_output)
        self.array_output = (1 / (1 + np.exp(-ex))) # sigmoid

        ### END ###

    def backward(self):
        # This function is for back propagation.
        ### EDIT HERE ###
        temp1 = (self.output_answer - self.array_output)
        temp1 *= self.array_output - self.array_output**2 #np.full(self.output_size, 1.0) - self.array_output**2
        tp = np.transpose(self.array_hidden.reshape(1, self.hidden_size))
        temp1 = temp1.reshape(1, self.output_size)
        result1 = np.dot(tp, temp1)
        self.temp_w_output = self.w_output + self.learning_rate*result1
 
        result2 = np.dot(temp1, np.transpose(self.w_output)) * (np.full(self.hidden_size, 1.0) - self.array_hidden**2).reshape(1, self.hidden_size)
        self.temp_w_input = self.w_input + self.learning_rate*np.dot(self.array_input.reshape(self.input_size, 1), result2)
        self.befor_error = self.error
        ### END ###

    def step(self):
        # This function is for weight updating.
        ### EDIT HERE ###
        self.w_input = self.temp_w_input
        self.w_output = self.temp_w_output
        ### END ###

    def loss(self):
        # This function is for calculating loss between logits and labels.
        ### EDIT HERE ###
        answer = float(0)
        temp = (self.output_answer - self.array_output)**2

        for i in range(self.output_size):
            answer += temp[i]
        answer /= 2
        self.error = answer
        ### END ###


def main(data, label2idx):
    ### EDIT HERE ###
    std_name = "이동혁"
    std_id = "2016311076"    
    ### END ###
    result = dict()
    for inp_type, tr, te in tqdm(data, desc='training & evaluating...'):

        result = dict()
        ### EDIT HERE ###
        mlp = MLP(len(tr[0][1]), 100, 5, 0.01)
        
        ## train
        before_errorSum = float(1000000) 
        errorSum = float(0)
        for k in range(100):
            for i in range(len(tr)):
                answer = np.full(5, 0.0)
                answer[tr[i][2]] = float(1)
                mlp.output_answer = answer
                mlp.forward(tr[i][1])
                mlp.loss()
                errorSum += mlp.error
                mlp.backward()
                mlp.step()
            if errorSum > before_errorSum:
                print("break")
                break;
            before_errorSum = errorSum
            errorSum = float(0)

        ## test
        for_score = np.zeros((5,2,2), dtype=np.float32)
        for_micro = np.zeros((5,5), dtype=np.float32)
        docs_cnt = np.zeros(5, dtype=np.int8)
        total_docs_cnt = 0

        prediction = list()
        answer_ = list()

        for i in range(len(te)):
            mlp.input_size = len(te[i][1])
            answer_list = np.full(5, 0.0)
            answer = te[i][2]
            answer_list[te[i][2]] = float(1)
            docs_cnt[answer] += 1
            total_docs_cnt += 1

            mlp.forward(te[i][1])
            predict = mlp.array_output
            predict_num = 0
            max_value = -1
            for j in range(5):
                if predict[j] > max_value:
                    predict_num=j
                    max_value = predict[j]

            for_micro[answer][predict_num] += 1
            prediction.append(predict_num)
            answer_.append(answer)

            if predict_num == answer:
                for j in range(5):
                    if j == predict_num:
                        for_score[j][0][0] += 1
                    else:
                        for_score[j][1][1] += 1
            else:
                for j in range(5):
                    if j == predict_num:
                        for_score[j][1][0] += 1
                    else:
                        for_score[j][0][1] += 1

        ## 점수
        print(answer_)
        print(prediction)

        accuracy = accuracy_score(answer_, prediction)
        precsion = precision_score(answer_, prediction, average='micro')
        recall = recall_score(answer_, prediction, average='micro')
        f1 = f1_score(answer_, prediction, average='micro')

        ma_precsion = precision_score(answer_, prediction, average='macro')
        ma_recall = recall_score(answer_, prediction, average='macro')
        ma_f1 = f1_score(answer_, prediction, average='macro')
    
        print("Accuracy: {}%".format(accuracy * 100))
        print("Precision: {}%".format(precsion * 100))
        print("Recall: {}%".format(recall * 100))
        print("F1: {}%".format(f1 * 100))

        print("ma_Precision: {}%".format(ma_precsion * 100))
        print("ma_Recall: {}%".format(ma_recall * 100))
        print("ma_F1: {}%".format(ma_f1 * 100))

        with open('./MLP_.txt', 'w', encoding='utf-8') as f:
            f.write("Accuracy: {}%\n".format(str(accuracy * 100)))
            f.write("Precision-micro: {}%\n".format(str(precsion * 100)))
            f.write("Recall-micro: {}%\n".format(str(recall * 100)))
            f.write("F1-micro: {}%\n\n".format(str(f1 * 100)))
            f.write("Precision-macro: {}%\n".format(str(ma_precsion * 100)))
            f.write("Recall-macro: {}%\n".format(str(ma_recall * 100)))
            f.write("F1-macro: {}%\n\n".format(str(ma_f1 * 100)))

        temp = dict()
        labels = ["entertainment", "finance", "lifestyle", "sports", "tv"]
        pres = np.zeros(5, dtype=np.float32)
        recalls = np.zeros(5, dtype=np.float32)
        for i in range(5):
            pres[i] = round((for_score[i][0][0]/(for_score[i][0][0]+for_score[i][1][0]))*100, 2)
            recalls[i] = round((for_score[i][0][0]/(for_score[i][0][0]+for_score[i][0][1]))*100, 2)

        for i in range(5):
            f1 = round(2*pres[i]*recalls[i]/(pres[i]+recalls[i]), 2)
            temp[labels[i]] = [pres[i], recalls[i], f1, docs_cnt[i]]

        micro_metrics = np.zeros((2,2), dtype=np.float32)
        macro_preds = np.zeros(5, dtype=np.float32)
        macro_recalls = np.zeros(5, dtype=np.float32)
        for i in range(5):
            temp_ = np.zeros((2,2), dtype=np.float32)

            temp_[0][0] += for_micro[i][i]
            for j in range(5):
                if i != j:
                    temp_[0][1] += for_micro[i][j]
            for j in range(5):
                if i != j:
                    temp_[1][0] += for_micro[j][i]
            for a in range(5):
                for b in range(5):
                    if a != i and b != i:
                        temp_[1][1] += for_micro[a][b]
            macro_preds[i] = round((temp_[0][0]/(temp_[0][0]+temp_[1][0]))*20, 2)
            macro_recalls[i] = round((temp_[0][0]/(temp_[0][0]+temp_[0][1]))*20, 2)
            for a in range(2):
                for b in range(2):
                    micro_metrics[a][b] += temp_[a][b]
        micro_pred = round((micro_metrics[0][0]/(micro_metrics[0][0]+micro_metrics[1][0]))*100, 2)        
        micro_recall = round((micro_metrics[0][0]/(micro_metrics[0][0]+micro_metrics[0][1]))*100, 2)
        micro_f1 = round(2*micro_pred*micro_recall/(micro_pred+micro_recall),2)
        macro_pred = float(0)
        macro_recall = float(0)
        macro_f1 = float(0)
        for i in range(5):
            macro_pred += macro_preds[i]
            macro_recall += macro_recalls[i]
            macro_f1 += round(2*macro_preds[i]*macro_recalls[i]/(macro_preds[i]+macro_recalls[i]),2)

        temp["micro avg"] = [round(micro_pred,2), round(micro_recall, 2), round(micro_f1, 2), len(te)]
        temp["macro avg"] = [round(macro_pred,2), round(macro_recall, 2), round(macro_f1, 2), len(te)]
        accuracy = round(((micro_metrics[0][0] + micro_metrics[1][1])/(micro_metrics[0][0]+micro_metrics[0][1]+micro_metrics[1][0]+micro_metrics[1][1]))*100,2)
        temp["accuracy"] = [accuracy, len(te)]

        result[inp_type] = temp


        ### END ###
    """
        result(input variable for "save_result" function) contains 
            1. Performance for each labels (precision, recall, f1-score per label)
            2. Overall micro average and accuracy for the entire test dataset
            3. Convert the result 1 and 2 into percentages by multiplying 100

        result type : Dict[str, Dict[str, Union[Tuple[float, float, float, int], Tuple[float, int]]]]
        result input format for "save_result" function: 
        {
            'Binary': 
            {
                "entertainment": (precision, recall, f1-score, # of docs),
                "finance": (precision, recall, f1-score, # of docs),
                "lifestyle": (precision, recall, f1-score, # of docs),
                "sports": (precision, recall, f1-score, # of docs),
                "tv": (precision, recall, f1-score, # of docs),
                "accuracy": (accuracy, total docs),
                "micro avg": (precision, recall, f1-score, total docs)
            },
            "TF": ...,
            "TF-IDF": ...,
        }
    """
    save_result(result, std_name=std_name, std_id=std_id)

if __name__ == "__main__":
    #   *** Do not modify the code below ***
    random.seed(42)
    np.random.seed(42)

    Preprocessing = Preprocessing()
    train_data = Preprocessing.load_data(data_path='./train.json', data_type='train')
    Preprocessing.tr_binary = Preprocessing.Calculate_Binary(data=train_data)
    Preprocessing.tr_tf = Preprocessing.Calculate_TF(data=train_data)
    Preprocessing.tr_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=train_data, data_type='train')
    test_data = Preprocessing.load_data(data_path='./test.json', data_type='test')
    Preprocessing.te_binary = Preprocessing.Calculate_Binary(data=test_data)
    Preprocessing.te_tf = Preprocessing.Calculate_TF(data=test_data)
    Preprocessing.te_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=test_data, data_type='test')

    data = [
        #('Binary', Preprocessing.tr_binary, Preprocessing.te_binary), 
        ('TF', Preprocessing.tr_tf, Preprocessing.te_tf), 
        #('TF-IDF', Preprocessing.tr_tfidf, Preprocessing.te_tfidf)
        ]

    main(data, Preprocessing.label2idx)
    #   *** Do not modify the code above ***
