# -*- coding: utf-8 -*-

import argparse
import json
import nltk
import numpy as np
import math
import random

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pathlib import Path
from sklearn.svm import *
from tqdm import tqdm, trange
from typing import Union, List, Dict, Tuple, Optional

nltk.download('punkt')                      
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class AI_util:
    def __init__(self):
        #   *** Do not modify the code ***
        # Noun, Verb POS tags
        self.extract_specific_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        # Stopword Set
        self.stop_words = set(stopwords.words('english'))
        self.word2idx: Dict[str, int]
        self.tr_binary: List[Tuple[str, List[float], int]]
        self.te_binary: List[Tuple[str, List[float], int]]
        self.tr_tf: List[Tuple[str, List[float], int]]
        self.te_tf: List[Tuple[str, List[float], int]]
        self.tr_tfidf: List[Tuple[str, List[float], int]]
        self.te_tfidf: List[Tuple[str, List[float], int]]
        self.label2idx = {label: idx for idx, label in enumerate(['entertainment', 'finance', 'lifestyle', 'sports', 'tv'])}
        #   *** Do not modify the code ***

    def load_data(self, data_path: Path, data_type: str = 'train') -> List[Tuple[str, List[str], int]]:
        #   *** Do not modify the code ***
        # Load input data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        t_data = list()
        t_token = set()
        for d in tqdm(data, desc='Tokenizing... : '):
            paragraph = d['paragraph']
            # tokenize and POS tagging
            pos_tokens = nltk.pos_tag(word_tokenize(paragraph.lower()))
            # Extract Noun, Verb
            tokenized_paragraph = [
                token[0] for token in pos_tokens
                if (token[1] in self.extract_specific_tags) and (not token[0] in self.stop_words)
            ]
            if data_type == 'train':
                t_token.update(tokenized_paragraph)
            t_data.append((d['id'], tokenized_paragraph, self.label2idx[d['label']]))
        # Create Train Vocab
        if data_type == 'train':
            self.word2idx = {word: idx for idx, word, in enumerate(sorted(t_token))}

        return t_data[:]
        #   *** Do not modify the code ***

    def Calculate_Binary(self):
        #   *** Do not modify the code ***
        return

    def Calculate_TF(self):
        #   *** Do not modify the code ***
        return

    def Calculate_TF_IDF_Normalization(self):
        #   *** Do not modify the code ***
        return

def save_result(result: Dict[str, Union[Tuple[float, float, float, int], Tuple[float, int]]], 
                std_name: Optional[str] = None, 
                std_id: Optional[str] = None):
    output_path = Path('./{}_{}.txt'.format(std_name, std_id))
    with open(output_path, mode='w', encoding="utf-8") as f:
        for inp_type in ['Binary', 'TF', 'TF-IDF']:
            tmp = result[inp_type]
            label_name = ['entertainment', 'finance', 'lifestyle', 'sports', 'tv']
            
            headers = ["precision", "recall", "f1-score", "# docs"]
            name_width = max(len(cn) for cn in label_name)
            width = max(name_width, len('micro avg'))
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            report = head_fmt.format('', *headers, width=width) + '\n\n'
            row_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9}\n'
            for label in label_name:
                report += row_fmt.format(
                    label, 
                    tmp[label][0], 
                    tmp[label][1], 
                    tmp[label][2], 
                    tmp[label][3], 
                    width=width
                )
            row_fmt_accuracy = '{:>{width}s} ' + \
                                ' {:>9.2}' * 2 + ' {:>9.2f}' + \
                                ' {:>9}\n'
            report += '\n' + row_fmt.format(
                'micro avg',
                tmp['micro avg'][0], 
                tmp['micro avg'][1], 
                tmp['micro avg'][2], 
                tmp['micro avg'][3], 
                width=width
            )
            report += row_fmt.format(
                'macro avg',
                tmp['macro avg'][0], 
                tmp['macro avg'][1], 
                tmp['macro avg'][2], 
                tmp['macro avg'][3], 
                width=width
            )
            report += row_fmt_accuracy.format(
                'accuracy', 
                '', 
                '', 
                tmp['accuracy'][0], 
                tmp['accuracy'][1], 
                width=width
            )

            f.write('Input Type : {}\n'.format(inp_type))
            f.write(report + '\n')
