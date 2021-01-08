
import numpy as np

import pandas as pd

import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.models import load_model

from keras.optimizers import Adadelta

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, LSTM, Concatenate, Lambda

import math
import re

import json
from flask import Flask, redirect, url_for, jsonify, request

app = Flask(__name__)

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def text_to_vector(inputlist, tokenizer):

  MAX_LENGTH = 20

  sequences = tokenizer.texts_to_sequences(inputlist)

    

  return pad_sequences(sequences, maxlen=MAX_LENGTH)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile("[^0-9a-z#+_]")


def clean_text(text):

  text = text.lower()

  text = REPLACE_BY_SPACE_RE.sub(' ', text)

  text = BAD_SYMBOLS_RE.sub(' ', text)

  text = text.replace('x','')

  return text

@app.route('/getsummary', methods=['GET'])
def getsummary():
    global model
    model = load_model('model_new.h5',compile=False)
    summary = model.name
    return summary

@app.route('/getJsonData', methods=['GET'])

def getJsonData():
    filedata = open('docs/Data.json',)
    jsondata = json.load(filedata)
    
    
    return json.dumps(jsondata);

@app.route("/hello")
def hello():
    return 'Hello World'
	
	
@app.route('/predictfunction',methods=['GET','POST'])            

def predictfunction(): 
    global model
    model = load_model('model_new.h5',compile=False)    
    TOKENIZER_PATH = 'tokenizer_new.pickle'
    
    
    tokenizer_pickle = open(TOKENIZER_PATH,"rb")
    tokenizer = pickle.load(tokenizer_pickle)
    
    posted_data = request.get_json()   
             
    store_list = []
    
    for item in posted_data['array']:
        store_list_sub = []
        store_list_sub.append(item['studentanswer'])
        store_list_sub.append(item['referenceanswer'])
        store_list.append(store_list_sub)
    #TEST_DATA_PATH = 'graded_answers_9k_2way_test.csv'
    
    raw_data = pd.DataFrame(store_list,columns=['studentanswer','referenceanswer'])
    raw_data['studentanswer'] = raw_data['studentanswer'].apply(clean_text)
    raw_data['referenceanswer'] = raw_data['referenceanswer'].apply(clean_text)
     
    
    reference_answer = raw_data['referenceanswer']
    student_answer = raw_data['studentanswer']
    
    
    reference_answer_vector = text_to_vector(reference_answer,tokenizer)
    student_answer_vector = text_to_vector(student_answer,tokenizer)
    
    predicted_result = model.predict([reference_answer_vector[:],student_answer_vector[:]])
    
    ctr = 0
    for i in predicted_result:
        ctr = ctr+1
        print(i)
        if ctr==5:
          break
    predicted_scores = []
    for i in predicted_result:
        k = int(np.round(i))
        if k==1:
          predicted_scores.append("correct")
        else:
          predicted_scores.append("incorrect")
    

    return json.dumps(predicted_scores)  



if __name__ == '__main__':
   app.run()