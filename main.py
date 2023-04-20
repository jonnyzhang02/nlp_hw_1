'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-04-10 08:52:32
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-04-20 15:41:39
FilePath: \nlp_hw_1\main.py
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
    # 将数字替换为#
    text = re.sub(r'[\d]', '#', text)
    return text


def init_vocab():
    global vocab # 词典
    for piece in corpus: # 语料库
        if len(piece): # 如果piece不为空
            for word in piece: # piece中的每个词汇 
                vocab[word] += 1 #  
    print('Vocab size: {}'.format(len(vocab))) # 词典大小
    return vocab

def get_frequency_merge_max():
    global corpus #     语料库
    global vocab # 词典
    pairs = collections.defaultdict(int) # 词对
    for piece in corpus: # 语料库
        for i in range(len(piece) - 1): # 每个piece中的词汇
            pairs[piece[i], piece[i + 1]] += 1 # 词对的频率
    # print('Pairs size: {}'.format(len(pairs)))
    sorted_pairs  = sorted(pairs.items(), key=lambda x: x[1], reverse=True) # 词对按照频率从大到小排序
    max_pair = sorted_pairs[0] # 词频最大的词对

    with open('./log.txt', 'a', encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        f.write("现在合并的词汇是" + str(max_pair) + '\n')
        f.write("词典大小" + str(len(vocab)) + '\n')
        f.write('\n')
    
    # 将vocab中的词汇进行合并
    merge_vocab(max_pair[0])

    # 将corpus中的词汇进行合并
    for j in range(len(corpus)): # 遍历语料库
        if len(corpus[j]) >= 2: # 如果piece中的词汇大于等于2
            for i in range(len(corpus[j]) - 1): # 遍历piece中的词汇
                if corpus[j][i] == max_pair[0][0] and corpus[j][i + 1] == max_pair[0][1]: # 如果词汇和最大词对中的词汇相同
                    corpus[j][i] = max_pair[0][0] + max_pair[0][1] # 将词汇合并
                    corpus[j][i+1] = '' # 将词汇合并
        corpus[j] = [s.strip() for s in corpus[j] if s.strip()] # 去除空格

    return pairs

def merge_vocab(pair):
    global vocab
    # 一开始以为是词典大小减小，笑了，全print才发现
    # print("现在合并的词汇是",pair,"词典大小",len(vocab))
    # print(vocab[pair[0]],vocab[pair[1]])
    # vocab[pair[0]+ pair[1]] = vocab[pair[0]] + vocab[pair[1]]
    # del vocab[pair[0]]
    # del vocab[pair[1]] 一开始以为是词典大小减小，笑了
    # print(vocab[pair[0]+ pair[1]])
    # print("合并后的词典大小",len(vocab),'\n')
    vocab[pair[0]+ pair[1]] = vocab[pair[0]] + vocab[pair[1]]



if __name__ == '__main__':
        init_corpus('./data/train_BPE.txt')
        init_vocab()
        while len(vocab) < 10000:
            get_frequency_merge_max()
            
            print((len(vocab)-5994)/(10000-5994)*100,"%")
            if len(vocab)%100 == 0:
                with open('./vocab.txt', 'w', encoding="utf-8") as f:
                    f.write(str(vocab))


    
