import numpy as np
import torch
import codecs
import math
import os

from bert_serving.client import BertClient
from parlai.core.dict import unescape

class BertEncoder():
        def __init__(self):
            self.bc = BertClient()
            self.vocab = {}

            
            
        def loadDict(self):
            i = 0
            filename = os.path.join('/userhome/student/bokanyi/transformer', 'transformer.dict')
            with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as read:
                for line in read:
                    split = line.strip().split('\t')
                    token = unescape(split[0])
                    self.vocab[i] = token
                    i = i + 1
            
           # i = 0
            #with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
             #   for index, line in enumerate(f):
              #      values = line.strip().split("\t")
                #    self.vocab[index] = values[0]
                 #for line in f:
                  #   split = line.strip().split("\t")
                   #  self.vocab[i] = split[0]
                    # i += 1
            
            
        def forward(self, input):
            sentences = np.ndarray(shape=(input.shape[0], input.shape[1], 768))
            
            for j in range(input.shape[0]):
                to_encode = []
                for i in range(input.shape[1]):
             
                    if self.vocab[input[j,i].item()] is not '__null__':
                        to_encode.append(self.vocab[input[j,i].item()])
                encoded = self.bc.encode(to_encode)
                null_size = sentences.shape[1] - encoded.shape[0]
                if null_size is not 0:
                    nulls = np.ndarray(shape=(null_size, 768))
                    sentences[j] = np.concatenate(encoded, nulls)
                else:
                    sentences[j] = encoded
            return torch.from_numpy(sentences)