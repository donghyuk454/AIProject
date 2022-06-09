# -*- coding: utf-8 -*-

from hw7_util import *
from sklearn.svm import LinearSVC
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

def main(data, label2idx):

    std_name = "이동혁"
    std_id = "2016311076"
    result = dict()
    for inp_type, tr, te in tqdm(data, desc='training & evaluating...'):
        """
            This function is for training and evaluating (testing) SVM Model.
        """
        ### EDIT HERE ###
        classifier = LinearSVC(C=1.0, max_iter=1000)
        train_inputs = list()
        train_labels = list()
        test_inputs = list()
        answer = list()
        for i in range(len(tr)):
            train_inputs.append(tr[i][1])
            train_labels.append(tr[i][2])

        classifier.fit(train_inputs, train_labels)
        for i in range(len(te)):
            test_inputs.append(te[i][1])
            answer.append(te[i][2])

        prediction = classifier.predict(test_inputs)

        for_score = np.zeros((5, 2,2), dtype=np.float32)
        for_micro = np.zeros((5,5), dtype=np.float32)
        docs_cnt = np.zeros(5, dtype=np.int8)
        
        for i in range(len(prediction)):
            if prediction[i] == answer[i]:
                for_score[prediction[i]][0][0] += 1
            else:
                for_score[prediction[i]][1][0] += 1
                for_score[answer[i]][0][1] += 1
            docs_cnt[answer[i]] += 1
            for_micro[answer[i]][prediction[i]] += 1
        
        for i in range(5):
            for_score[i][1][1] = len(prediction) - for_score[i][0][0] - for_score[i][1][0] - for_score[i][0][1] 

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

        accuracy = accuracy_score(answer, prediction)
        precsion = precision_score(answer, prediction, average='micro')
        recall = recall_score(answer, prediction, average='micro')
        f1 = f1_score(answer, prediction, average='micro')

        ma_precsion = precision_score(answer, prediction, average='macro')
        ma_recall = recall_score(answer, prediction, average='macro')
        ma_f1 = f1_score(answer, prediction, average='macro')
    
        print("Accuracy: {}%".format(accuracy * 100))
        print("Precision: {}%".format(precsion * 100))
        print("Recall: {}%".format(recall * 100))
        print("F1: {}%".format(f1 * 100))

        print("ma_Precision: {}%".format(ma_precsion * 100))
        print("ma_Recall: {}%".format(ma_recall * 100))
        print("ma_F1: {}%".format(ma_f1 * 100))

        with open('./SVM_.txt', 'w', encoding='utf-8') as f:
            f.write("Accuracy: {}%\n".format(str(accuracy * 100)))
            f.write("Precision-micro: {}%\n".format(str(precsion * 100)))
            f.write("Recall-micro: {}%\n".format(str(recall * 100)))
            f.write("F1-micro: {}%\n\n".format(str(f1 * 100)))
            f.write("Precision-macro: {}%\n".format(str(ma_precsion * 100)))
            f.write("Recall-macro: {}%\n".format(str(ma_recall * 100)))
            f.write("F1-macro: {}%\n\n".format(str(ma_f1 * 100)))
        
        ### EDIT FROM HERE ###
        """ Implement codes for the output text file. """
        '''length = len(prediction)
        for i in range(0, length):
            if prediction[i] == 0:
                f.write("{}\tfinance\t{}\n".format(str(ids[i]), str(values[i])))
'''
        ### END ###

    save_result(result, std_name=std_name, std_id=std_id)

if __name__ == "__main__":
    #   *** Do not modify the code below ***
    random.seed(42)
    np.random.seed(42)

    Preprocessing = Preprocessing()
    tr_data = Preprocessing.load_data(data_path='./train.json', data_type='train')
    Preprocessing.tr_binary = Preprocessing.Calculate_Binary(data=tr_data)
    Preprocessing.tr_tf = Preprocessing.Calculate_TF(data=tr_data)
    Preprocessing.tr_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=tr_data, data_type='train')
    te_data = Preprocessing.load_data(data_path='./test.json', data_type='test')
    Preprocessing.te_binary = Preprocessing.Calculate_Binary(data=te_data)
    Preprocessing.te_tf = Preprocessing.Calculate_TF(data=te_data)
    Preprocessing.te_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=te_data, data_type='test')

    data = [
        #('Binary', Preprocessing.tr_binary, Preprocessing.te_binary), 
        ('TF', Preprocessing.tr_tf, Preprocessing.te_tf), 
        #('TF-IDF', Preprocessing.tr_tfidf, Preprocessing.te_tfidf)
        ]

    main(data, Preprocessing.label2idx)
    #   *** Do not modify the code above ***