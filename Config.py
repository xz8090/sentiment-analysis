# -*- coding: utf-8 -*-

#定义参数
SENTENCE_NUM = 10000
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2


#训练
epochs = 10
batch_size = 50 
#choice : BIGRU/BILSTM/MLP/ATENTION_LSTM/ATTENTION_GRU   
choice = 'ATTENTION_GRU'


#预测
model_file = './ATTENTION_GRU_model/ATTENTION_GRU.bin'
#model_file = './lstm_model/lstm.bin'
#model_file = './ATENTION_LSTM_model/ATENTION_LSTM.bin'

string_list = ['北化环境很美尽管地方不大',
               '天气很好，非常开心',
               '在这家店买的东西质量很差，一点诚信都没有，不会再光顾了']

              