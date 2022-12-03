import os
import numpy as np
import psutil
from typing import Optional, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from transforms import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ref_doc_path: str, transformation):
        self.documents = [
            os.path.join(ref_doc_path, path)
            for path in os.listdir(ref_doc_path)
            if path.endswith((".csv"))
        ]

        self.transform = transform
        self.text_array = np.empty(0)
        self.label_array = np.empty(0)
        self.setup()
        #Vocab
        self.label_encoder = LabelEncoder()
        #self.label_encoder.fit(self.reference_labels)

    def __len__(self) -> int:
        return len(self.text_array)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
        return (self.text_array[idx],self.label_array[idx])

    def setup(self):
        #text_array = np.empty(0,0)
        #label_array = np.empty(0,0)
        for doc in self.documents:
            self.text_array = np.append(self.text_array,np.loadtxt(doc,delimiter=',',quotechar='"',dtype=str,encoding='utf8',usecols=(0)))
                #Label column may need changing
            self.label_array = np.append(self.label_array,np.loadtxt(doc,delimiter=',',quotechar='"',dtype=str,encoding='utf8',usecols=(1)))
        #Train/Val/Test split
        return(self.text_array,self.label_array)

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    def build_vocabulary(self):
        vocab = build_vocab_from_iterator(np.nditer(self.text_array))
        return(vocab)