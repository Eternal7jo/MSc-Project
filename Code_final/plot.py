#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import matplotlib.pyplot as plt
import argparse
import numpy as np
# fig3 = plt.figure(figsize=(20, 15))
# plt.plot(np.arange(train_size+1, len(dataset)+1, 1), scaler.inverse_transform(dataset)[train_size:], label='dataset')
# plt.plot(testPredictPlot, 'g', label='test')
# plt.ylabel('price')
# plt.xlabel('date')
# plt.legend()
# plt.show()


def draw_curve(data, train_loss=False, train_mrr=False, valid_mrr=False):
    if train_loss:
        tmp = 'train_loss'
    elif train_mrr:
        tmp = 'train_mrr'
    elif valid_mrr:
        tmp = 'valid_mrr'

    fig1 = plt.figure(figsize=(12, 8))
    plt.plot(data)
    print(data)
    plt.title(f'{tmp}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def main():

    parser = argparse.ArgumentParser(description='translate_1.py')

    # parser.add_argument('-model', required=True,
    #                     help='Path to model weight file')
    # parser.add_argument('-data_pkl', required=True,
    #                     help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-file_path', default='save/models/transformer/python-2021-07-19_02-59/results.txt', help='Path of log file')

    args = parser.parse_args()
    filename = args.file_path

    train_loss = []
    train_mrr = []
    valid_mrr = []
    data = open(filename, 'r', encoding='utf-8').readlines()
    data_pre = data[1:-1]
    for line in data_pre:
        train_loss_line, train_mrr_line, _, valid_mrr_line = line.split('\t')
        train_loss_line = round(float(train_loss_line), 4)
        train_mrr_line = round(float(train_mrr_line), 4)
        valid_mrr_line = round(float(valid_mrr_line), 4)

        train_loss.append(train_loss_line)
        train_mrr.append(train_mrr_line)
        valid_mrr.append(valid_mrr_line)

    draw_curve(train_loss, train_loss=True)
    draw_curve(train_mrr, train_mrr=True)
    draw_curve(valid_mrr, valid_mrr=True)
    print('Done!')


if __name__=='__main__':
    main()