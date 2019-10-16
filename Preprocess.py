# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:42:43 2018

@author: 29oct
"""
from random import randint
import pandas as pd
import torch
from InferSent import models
import nltk,string
import numpy as np
from allennlp.modules.elmo  import Elmo, batch_to_ids
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer,sent_tokenize
from nltk.corpus import state_union,stopwords, wordnet
print("preprocess.py")
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


# nltk.download('state_union')
# nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

def getWordnetPOS(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# tokenizer without preprocessing#
def tokenizer(doc_text):
    sentences=sent_tokenize(doc_text)
    tt = ToktokTokenizer()
    allTokens = []
    for i in sentences:
        tokens = tt.tokenize(i)
        for token in tokens:
            if(all(i in string.punctuation for i in token)):
                tokens.remove(token)
        allTokens.extend(tokens)
    return allTokens  

def preprocess(text):
        sentences=sent_tokenize(text)
        tokenizer = ToktokTokenizer()
        text_lemmatized_words=[]
        for i in sentences:
            tokens = tokenizer.tokenize(i)
            tokens = [token.lower() for token in tokens if(len(token)>1)] #removing single char tokens
            for word in tokens:
                if(all(i in string.punctuation for i in word)):
                    tokens.remove(word)
            tags = nltk.pos_tag(tokens)
            stop_words = set(stopwords.words('english'))
            processedWords = [token for token in tags if token[0] not in stop_words]
            lemmatizedWords = [lemmatizer.lemmatize(word[0], pos=getWordnetPOS(word[1])) for word in processedWords]  
            text_lemmatized_words.extend(lemmatizedWords) 
        return text_lemmatized_words 

def getUniversalSentenceEncoding(text):
    print(text)
    text=str(text)
    model_version = 1
    MODEL_PATH = "InferSent/infersent1.pkl"
    params_model = {'bsize': 1, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = models.InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    use_cuda = True
    model = model.cuda() if use_cuda else model
    W2V_PATH = 'InferSent/glove.840B.300d.txt' #if model_version == 1 else '../dataset/fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)
    claims=[text]
    embedding = model.encode(claims, bsize=1, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(embedding)))
    return embedding[0]


#ELMo word embeddings#
def getContextualisedWordEmbeddings(tokens):    
    elmo = Elmo('/media/sdb/sanjay/IR/debunking-fake-news/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json', 
    '/media/sdb/sanjay/IR/debunking-fake-news/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', 2, dropout=0)
    sentences = [tokens]
    print("no of sentences"+str(len(tokens)))
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    embeddings = embeddings['elmo_representations'][0][0]
    #print(embeddings)
    return embeddings
# from allennlp.commands.elmo import ElmoEmbedder

# def getContextualisedWordEmbeddings2(tokens):    
#     elmo = ElmoEmbedder()    
#     vectors = elmo.embed_sentence(tokens)
#     embeddings = vectors[2]
#     #print(embeddings)
#     return embeddings
#a=getUniversalSentenceEncoding("obama was born in madhya pradesh")
#b=getContextualisedWordEmbeddings(['obama','born', 'madhya pradesh'])
# # print(a)
# # print(a.shape)
# # print(type(a))
# print(b.shape)
# print(type(b))
# print(len(b))