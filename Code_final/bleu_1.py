#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(reference=None, translation=None):
    if reference and translation:
        reference = reference.split('\n')
        translation = translation.split('\n')
        inf_tok = [[nltk.tokenize.word_tokenize(i)] for i in reference]
        tra_tok = [nltk.tokenize.word_tokenize(t) for t in translation]
        chencherry =  SmoothingFunction()
        return corpus_bleu(inf_tok, tra_tok, smoothing_function=chencherry.method7)
    else:
        return "No files !"

if __name__ == '__main__':
    text = 'Today is a good day'
    gold = 'Today a good day'
    res = compute_bleu(text, gold)
    print(res)