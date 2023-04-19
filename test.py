'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-04-18 09:27:51
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-04-19 11:15:26
FilePath: \nlp_hw_1\test.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangyang0207@bupt.edu.cn, All Rights Reserved. 
'''
import re
import time
vocab = {}

def load_vocab():
    global vocab
    # 打开文件并读取所有行
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 将所有行合并为一行
    merged_line = ''.join(lines)

    vocab = eval(merged_line)


def BPE_split(sentence):
    global vocab
    # 从后往前遍历句子
    for i in range(len(sentence)-1, 0, -1):
        # 将句子中的两个字符组成一个新的字符
        new_char = sentence[i-1] + sentence[i]
        # 如果新的字符在词典中，则将新的字符替换原来的两个字符
        if new_char in vocab:
            sentence = sentence.replace(new_char, ' /'+new_char+'\\ ')
            # 从头开始遍历
            i = 0

    return sentence


def read_test_file():
    with open('./data/test_BPE.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open('./test_BPE_result.txt', 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            sentences = re.split(r'([, 。？！；’ ‘ “ ” 、])', line)
            i = 0
            while i < len(sentences) - 1:
                f.write(BPE_split(sentences[i])+sentences[i+1]+'\n')
                i = i+2

load_vocab()
read_test_file()
