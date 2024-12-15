import requests
import torch

from collections import defaultdict
from string import punctuation

from torch import Tensor
from torch.utils.data import Dataset


try:
  from torchtext.data import get_tokenizer
except:
  try:
    import nltk
    nltk.download('punkt_tab')
    def get_tokenizer(param):
      return nltk.word_tokenize
  except:
    print("ERROR:", "No tokenizer found!")

def count_parameters(m):
  psum = 0
  for p in m.parameters():
    psum += p.numel()
  return f"{psum:,}"

class TextUtils():
  stop_1000_url = "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
  stop_100_url  = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"

  stopwords_list = requests.get(stop_100_url).content
  stopwords = list(set(stopwords_list.decode().splitlines()))

  @staticmethod
  def create_vocab(text, max_words=200_000):
    # create one big string
    if type(text) is list:
      text = " ".join(text)

    # remove punctuation, whitespaces and convert to lowercase
    text = text.translate(str.maketrans('', '', punctuation)).lower().strip()

    # tokenize words
    tokenizer = get_tokenizer("basic_english")
    words = tokenizer("".join(text))[:max_words]

    # remove repeated words
    vocab = list(set(words))

    return words, vocab

class TextSequenceDataset(Dataset):
  def __init__(self, text, max_words=200_000, window=2, symmetric_context=True):
    super().__init__()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.window = window
    self.symmetric_context = symmetric_context
    self.words, self.vocab = TextUtils.create_vocab(text, max_words=max_words)
    wtoi = {word: i for i, word in enumerate(["<UNK>"] + self.vocab)}
    itow = {i: word for i, word in enumerate(wtoi)}

    self.wtoi = defaultdict(int, wtoi)
    self.itow = defaultdict(lambda: "<UNK>", itow)
    print(f"{len(self.wtoi)} words in vocab")
    print(f"{len(self.words)} words in text")

  def encode_word(self, word, return_tensors=False):
    widx = self.wtoi[word.lower()]
    if not return_tensors:
      return widx
    else:
      return Tensor([widx]).long().to(self.device)

  def encode(self, words):
    widx = [self.wtoi[w.lower()] for w in words]
    return Tensor(widx).long().to(self.device)

  def decode_word(self, idx):
    if type(idx) is int:
      return self.itow[idx]
    else:
      return self.itow[idx.item()]

  def decode(self, idxs_t):
    idxs = idxs_t.tolist()
    return [self.itow[i] for i in idxs]