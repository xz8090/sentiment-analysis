# -*- coding: utf-8 -*-
from scipy.misc import imread
import jieba
import json
import urllib
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from Attention_layer import Attention_layer
from Config import SENTENCE_NUM,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM,VALIDATION_SPLIT
from Config import model_file,string_list
from flask import Flask
import Spider

app = Flask(__name__)
model_file = './ATTENTION_GRU_model/ATTENTION_GRU.bin'
print ('loading model......')
model = load_model(model_file,{'Attention_layer':Attention_layer})

#对句子经行分词，并去掉换行符
def split_sentence(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

#预测
def predict_result(model, string):
    tx = [string]
    txs = split_sentence(tx)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(txs)
    sequences = tokenizer.texts_to_sequences(txs)
#    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = model.predict(data)
    result_0 = result[0][0]
    result_1 = result[0][1]
    return result_0, result_1

def get_mask():
    x,y = np.ogrid[:300,:300]
    mask = (x-150) ** 2 + (y-150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    return mask

def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
        h  = np.random.randint(0,200)
        s = np.random.randint(60,100)
        l = np.random.randint(50,100)
        return "hsl({}, {}%, {}%)".format(h, s, l)

@app.route('/getWordCloud/<num>/<text>',methods=['GET', 'POST'])
def get_word_cloud(num,text):
    print(num)
    text = urllib.parse.unquote(text)
    print(text)
    text = json.loads(text)
    wc = WordCloud(font_path = r'./data/font/huawen.ttf',background_color='#2C344B',mask=get_mask(), color_func = random_color_func)
    #wc = WordCloud(font_path = r'./data/font/huawen.ttf',background_color=None,mask=imread('Chinese.png'))
    wc.generate_from_frequencies(text)
    picPath = "D:\\Documents\\WeChat Files\\txt\\test"+num+".png"
    wc.to_file(picPath)
    return picPath
    
@app.route('/modelAPI/<text>',methods=['GET', 'POST'])
def start_work(text):
    """
    添加自定义损失或者网络层
    tips:
    load_model函数提供了custom_objects参数，所以加载时需要加入这个参数
    
    假设自定义参数loss的函数名为cosloss,所以加载时应采用以下方式
    from * import cosloss
    model = load_model(model_file, {'cosloss':cosloss})
    
    假设自定义网络层的函数名为Attention_layer,所以加载时应采用以下方式
    from Attention_layer import Attention_layer
    model = load_model(model_file,{'Attention_layer':Attention_layer})
    """
    text = urllib.parse.unquote(text)
    print(text)
    
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',metrics=['acc'])
    # model.summary()
    print('--------------------------------')
    print('预测结果')

    #for string in string_list:
    result_0, result_1 = predict_result(model, text)
    if result_0 > result_1:
        print('这段文字预测为0的概率为{}'.format(result_0))
        return json.dumps({'text':text,'confidence':str(result_0),'type':'0'},ensure_ascii = False)
    else:
        print('这段文字预测为1的概率为{}'.format(result_1))
        return json.dumps({'text':text,'confidence':str(result_1),'type':'1'},ensure_ascii = False)

@app.route('/getCommentsAPI/<id>/<page>',methods=['GET', 'POST'])
def getComments(id,page):
    id = str(id)
    page = str(page)
    dataList = Spider.getHotelComments(id,page)
    return dataList
    #return json.dumps(dataList)
    
if __name__ == '__main__':
    app.run()
    