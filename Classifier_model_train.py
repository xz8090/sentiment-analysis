# -*- coding: utf-8 -*-

import os
import jieba
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from Config import epochs,batch_size,choice
from Config import SENTENCE_NUM,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM,VALIDATION_SPLIT
from Utils import model_select
import keras
import matplotlib.pyplot as plt
#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        
#加载训练文件
def loadfile():
    neg=pd.read_excel('./data/tsb/neg.xls',header=None,index=None)
    pos=pd.read_excel('./data/tsb/pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))#1记为pos，0记为neg

    return combined,y

#对句子经行分词，并去掉换行符
def split_sentence(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


#读取词向量，生成词典
def embedding_dict():
    embeddings_index = {}
    f = open('./data/word2vec.word2vec',encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index


#补齐数据维度
def data_pad(texts):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)#将字或者词组转化成数字
    tokenizer.fit_on_texts(texts)#用作训练的文本序列
    sequences = tokenizer.texts_to_sequences(texts)#转化为序列
    word_index = tokenizer.word_index#序列索引
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #长度不同的序列进行0填充
    return data,word_index


#构造数据
def data_classfier():
    combined,y = loadfile()#combined是中文文本，y是极性（0是负，1是正）
    print(combined)
    texts = split_sentence(combined)#文本分词并去掉换行符
    labels = y
    labels = to_categorical(np.asarray(labels))#将y向量转为二进制矩阵
    print('text len', len(texts))
    print('labels len', len(labels))

    data,word_index = data_pad(texts)#词转序列

    indices = np.arange(data.shape[0])#取词序列中的第一维数据作为索引
    np.random.shuffle(indices)#打乱每个评论中词序列中的元素
    data = data[indices]
    labels = labels[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])#按比例VALIDATION_SPLIT抽取验证集数量，这里取0.2
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train,y_train,x_val,y_val,word_index

#模型选择
def model_(word_index,embeddings_index,choice):
    M = model_select()
    log_dir = './{}_model/log'.format(choice) 
    filepath = './{}_model/{}.h5'.format(choice,choice)
    if choice == 'BIGRU':
        model = M.BIGRU_model(word_index,embeddings_index)
    elif choice == 'BILSTM':
        model = M.BILSTM_model(word_index,embeddings_index)
    elif choice == 'MLP':
        model = M.MLP_model(word_index,embeddings_index)
    elif choice == 'ATENTION_LSTM':
        model = M.ATENTION_LSTM_model(word_index,embeddings_index)
    elif choice == 'ATTENTION_GRU':
        model = M.ATENTION_GRU_model(word_index,embeddings_index)
    else:
        print('选择的模型未存在可选配置中，请选择BIGRU/BILSTM/MLP/ATENTION_LSTM/ATTENTION_GRU中的一个')
        #终止程序， os._exit(0) 正常退出
#        os._exit(0)
    return log_dir,filepath,model

def mdk(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

#构造模型，训练
def train(epochs,batch_size,choise):    
    x_train,y_train,x_val,y_val,word_index = data_classfier()
    embeddings_index = embedding_dict()
    log_dir,filepath,model = model_(word_index,embeddings_index,choise)
    print(log_dir, filepath)
    mdk(log_dir)

    print('训练集和测试集正负面评论数量')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    tensorboard = TensorBoard(log_dir=log_dir)
    #保存最优模型
    checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',mode='max' ,save_best_only='True')
    #创建一个实例history
    #history = LossHistory()
    callback_lists=[tensorboard,checkpoint]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callback_lists)
   
    #测试集
    score = model.evaluate(x_val, y_val, batch_size=batch_size)
    print('loss: {}    acc: {}'.format(score[0], score[1]))
    
    #绘制acc-loss曲线
    #history.loss_plot('epoch')

if __name__ == '__main__':
    epochs = epochs   
    batch_size = batch_size
    choice = choice
    train(epochs,batch_size,choice)
    
    