import unicodedata
import string
import re
import random
import torch
from nltk import word_tokenize as tokenize
from parameter import *


class Corpus:
    def __init__(self, name, file_path=''):
        self.name = name  # Record language name.
        self.file_path = file_path  # Corpus file path of language.
        self.lines = []  # Content of corpus.
        self.n_lines = 0   # Number of lines of corpus

        # Read the file and split into lines.
        if self.file_path == '':
            self.file_path = DATA_PATH + '\\' + name + '.txt'
        self.lines = open(self.file_path,
                          encoding='UTF-8').read().strip().split('\n')

        # Format the lines.
        self.lines = [normalize_string(l) for l in self.lines]
        self.n_lines = len(self.lines)


# This class keep corpus's lang, word2index, word2count, index2word info.
class Lang:
    def __init__(self, name, file_path=""):
        self.name = name  # Record the language name.
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}

    # Init language model from corpus
    def load_corpus(self, corpus):
        self.n_words = 3
        lines = corpus.lines
        self.word2index = {PAD_word: 0, SOS_word: 1, EOS_word: 2, UNK_word: 3}
        self.word2count = {}
        self.index2word = {0: PAD_word, 1: SOS_word, 2: EOS_word, 3: UNK_word}
        for l in lines:
            # for word in tokenize(l):
            for word in l.split(' '):
                self.index_word(word)
                self.count_word(word)

    # Caculate word index, word count from a sentence.
    def cal_lines(self):
        for l in self.lines:
            # for word in tokenize(l):
            for word in l.split(' '):
                self.index_word(word)
                self.count_word(word)

    # Index one word.
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    # Count one word.
    def count_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1


# Convert encode?
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    # s = unicode_to_ascii(s.lower().strip())
    s = s.lower()
    s = re.sub(r"([,.!?])", r" \1 ", s)
    # s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Remove words below a certain count threshold
def trim_vocab(lang, min_count):
    word2count1 = {}
    vocab_size0 = len(lang.word2count)
    for k, v in lang.word2count.items():
        if v >= min_count:
            word2count1[k] = v
    lang.word2count = word2count1
    vocab_size1 = len(lang.word2count)

    # Reinitialize dictionaries after trim
    lang.word2index = {PAD_word: 0, SOS_word: 1, EOS_word: 2, UNK_word: 3}
    lang.index2word = {0: PAD_word, 1: SOS_word, 2: EOS_word, 3: UNK_word}
    lang.n_words = 3
    for word in lang.word2count.keys():
        lang.word2index[word] = lang.n_words
        lang.index2word[lang.n_words] = word
        lang.n_words += 1

    return vocab_size0, vocab_size1


# Filtering pairs by len
def filter_sentlen(corpus1, corpus2):
    lines1 = corpus1.lines
    lines2 = corpus2.lines
    keep_lines1 = []
    keep_lines2 = []
    for l1, l2 in zip(lines1, lines2):
        if len(l1) >= MIN_LENGTH and\
            len(l1) <= MAX_LENGTH and\
            len(l2) >= MIN_LENGTH and\
           len(l2) <= MAX_LENGTH:
            keep_lines1.append(l1)
            keep_lines2.append(l2)
    corpus1.n_lines = len(keep_lines1)
    corpus2.n_lines = len(keep_lines2)
    corpus1.lines = keep_lines1
    corpus2.lines = keep_lines2
    return corpus1.n_lines, corpus2.n_lines


# Filtering pairs by unknow word
def filter_unkword(corpus1, corpus2, lang1, lang2):
    lines1 = corpus1.lines
    lines2 = corpus2.lines
    keep_lines1 = []
    keep_lines2 = []
    for l1, l2 in zip(lines1, lines2):
        keep1 = True
        keep2 = True

        for w in l1.split(' '):
            if w not in lang1.word2index:
                keep1 = False
                break

        for w in l2.split(' '):
            if w not in lang2.word2index:
                keep2 = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep1 and keep2:
            keep_lines1.append(l1)
            keep_lines2.append(l2)

    corpus1.n_lines = len(keep_lines1)
    corpus2.n_lines = len(keep_lines2)
    corpus1.lines = keep_lines1
    corpus2.lines = keep_lines2
    return corpus1.n_lines, corpus2.n_lines


# Return a list of indexes, one for each word in the sentence, plus EOS
def indices_from_sentence(lang, sentence):
    # return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_index]
    return [lang.word2index.get(word, 3) for word in sentence.split(' ')] + [EOS_index]


# Return a list of indexes, one for each word in the sentence, plus EOS
def indices_to_sentence(lang, sindices):
    if isinstance(sindices, torch.Tensor):  # input is torch var
        sindices = sindices.squeeze()
        if sindices.is_cuda:    # input is gpu torch var
            sindices = sindices.cpu()
        sindices = sindices.numpy().tolist()
    return " ".join([lang.index2word[index] for index in sindices])


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_index for _ in range(max_length - len(seq))]
    return seq


# Random select pair lines
def random_pair(lines1, lines2, n):
    index = random.randint(0, n - 1)
    pair = [lines1[index], lines2[index]]
    return pair


def to_batch(lang, line):
    line = normalize_string(line)
    line = indices_from_sentence(lang, line)
    line = [line]
    length = [len(line)]
    var = torch.LongTensor(line).transpose(0, 1)
    if USE_CUDA:
        var = var.cuda()
    return var, length


# Pick random batch and pad
def random_batch(corpus1, corpus2, lang1, lang2, batch_size):
    in_seqs = []
    tar_seqs = []

    # Choose random pairs
    for _ in range(batch_size):
        pair = random_pair(corpus1.lines, corpus2.lines, corpus1.n_lines)
        in_seqs.append(indices_from_sentence(lang1, pair[0]))
        tar_seqs.append(indices_from_sentence(lang2, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(in_seqs, tar_seqs),
                       key=lambda p: len(p[0]), reverse=True)
    in_seqs, tar_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to
    # max length
    in_lengths = [len(s) for s in in_seqs]
    in_padded = [pad_seq(s, max(in_lengths)) for s in in_seqs]
    tar_lengths = [len(s) for s in tar_seqs]
    tar_padded = [pad_seq(s, max(tar_lengths)) for s in tar_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into
    # (max_len x batch_size)
    in_var = torch.LongTensor(in_padded).transpose(0, 1)
    tar_var = torch.LongTensor(tar_padded).transpose(0, 1)

    if USE_CUDA:
        in_var = in_var.cuda()
        tar_var = tar_var.cuda()

    return in_var, in_lengths, tar_var, tar_lengths
