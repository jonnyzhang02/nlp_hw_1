'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-04-10 08:52:32
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-04-18 09:02:50
FilePath: \nlp_hw_1\old.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangyang0207@bupt.edu.cn, All Rights Reserved. 
'''
import re
import collections
import time

corpus = [] #语料 一个二维数组，每个元素是一个piece，piece是一个一维数组，每个元素是一个词汇
vocab = collections.defaultdict(int) # 词典

def init_corpus(txtfile):
    # 初始化语料库
    """
    读取txt文件，将标点符号转换为#，并以空格分割，将每个词汇作为一个piece，将每个piece作为一个训练数据
    """
    piece = []
    with open(txtfile, 'r', encoding="utf-8") as f:  # 读取文件
        for line in f: # 读取每一行
            line = punctuation_removal(line) # 将标点符号转换为#
            words = line.split(' ') # 以空格分割
            for word in words:
                if word == '#' or word == '#\n': 
                    #如果是标点符号分割开的，或者不同行的，必然不会是一个词汇，所以把他们放在不同的piece里面
                    corpus.append(piece)
                    # piece 作为一条训练数据你数据
                    # print(piece,"\n")
                    # time.sleep(1)
                    piece = []
                else:
                    piece.append(word)
    print(len(corpus))
    return corpus


def punctuation_removal(text):
    # 将中文标点符号替换为#
    text = re.sub(r"[\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F]", '#', text)
    # 将英文标点符号替换为#
    text = re.sub(r"[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]", '#', text)
    return text


def init_vocab():
    global vocab
    for piece in corpus:
        for word in piece:
            vocab[''.join(list(word)) + '</w>'] += 1 # 词典
    print('Vocab size: {}'.format(len(vocab))) # 词典大小
    return vocab

def init_frequency():
    global corpus
    pairs = collections.defaultdict(int)
    for piece in corpus:
        for i in range(len(piece) - 1):
            pairs[piece[i], piece[i + 1]] += 1
    print('Pairs size: {}'.format(len(pairs)))
    sorted_pairs  = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    for pair in sorted_pairs:
        print(pair)
        time.sleep(1)
    return pairs


if __name__ == '__main__':
        init_corpus('./data/train_BPE.txt')
        init_vocab()
        init_frequency()


    
