'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-04-10 08:52:32
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-04-11 20:34:32
FilePath: \nlp_hw_1\main.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangyang0207@bupt.edu.cn, All Rights Reserved. 
'''
import os
import re
import sys
import pickle
import collections
from tqdm import tqdm

def punctuation_removal(text):
    # 将中文标点符号替换为#
    text = re.sub(r'[，。？！、‘’“”【】（）《》：；]', '#', text)
    # 将英文标点符号替换为#
    text = re.sub(r'[,.?!\'\"(){}:;]', '#', text)
    return text

def get_vocab(filename):
    vocab = collections.defaultdict(int) # 词典
    with open(filename, 'r', encoding="utf-8") as f:  # 读取文件
        for line in f: # 读取每一行
            line = punctuation_removal(line) # 去除标点符号
            words = line.split('\n') # 以空格分割
            for word in words:
                vocab[''.join(list(word.strip())) + ' </w>'] += 1 # 以</w>结尾
    print('Vocab size: {}'.format(len(vocab)))
    return vocab

def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq  
    print(tokens_frequencies)
    print('Tokens size: {}'.format(len(tokens_frequencies)))
    return tokens_frequencies

def get_stats(vocab):
    # 统计相邻词频
    pairs = collections.defaultdict(int) # 相邻词频
    for word, freq in vocab.items(): # 遍历词典     
        symbols = word.split() # 以空格分割
        for i in range(len(symbols)-1): # 遍历每个词
            pairs[symbols[i],symbols[i+1]] += freq # 统计相邻词频
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)
    
def get_sorted_tokens(filename, k):
    vocab = get_vocab(filename)
    tokens_frequencies = get_tokens_from_vocab(vocab)

    while len(tokens_frequencies) < k:
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        tokens_frequencies[''.join(best)] += 1

    sorted_tokens_tuple = sorted(
                            tokens_frequencies.items(), 
                            key=lambda item: (measure_token_length(item[0]), item[1]), 
                            reverse=True
                        )
    return sorted_tokens_tuple

def train(filename, k):
    vocab = get_vocab(filename) # get vocab
    tokens_frequencies = get_tokens_from_vocab(vocab)

    while len(tokens_frequencies) < k:
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        tokens_frequencies[''.join(best)] += 1

    sorted_tokens_tuple = sorted(
                            tokens_frequencies.items(), 
                            key=lambda item: (measure_token_length(item[0]), item[1]), 
                            reverse=True
                        )
    F = open('sorted_tokens','wb')
    pickle.dump(sorted_tokens_tuple, F)
    F.close()

if __name__ == '__main__':
        train('./data/train_BPE.txt', 10000)


    
