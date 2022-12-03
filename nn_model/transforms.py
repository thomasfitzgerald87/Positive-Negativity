import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator

strTest = 'Hello, this is my paragraph.  I wrote...words in it.  They are pretty great:would you not agree?'
strListTest = ['Hello, this is my list.','I wrote multiple sentences in it.','They are pretty bad.']
VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"

basic_tokenizer = get_tokenizer('basic_english')
#BERT_tokenizer = torchtext.transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)

#print(BERT_tokenizer(strTest))
#print(BERT_tokenizer(strListTest))

class transform(object):
    def __init__(self):
        self.transform = get_tokenizer('basic_english')