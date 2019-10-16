import numpy
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import TrainTestSplit
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
class Attn(nn.Module):
    def __init__(self, method,feature_size_1,feature_size_2,hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.feature_size_1 = feature_size_1
        self.feature_size_2 = feature_size_2
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.linear = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.linear = nn.Linear(self.feature_size_1+self.feature_size_2 , hidden_size)
            self.parameter = nn.Parameter(torch.FloatTensor(1, hidden_size))
            # self.tanh=nn.tanh()

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        hidden = hidden.unsqueeze(0) 
        
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()
            #print ("attn_energies.shape in attention"+str(attn_energies.shape))
        # Calculate energies for each encoder output
        for i in range(seq_len):
            #print (i)
            #print ("hidden.shape in attention"+str(hidden.shape))
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0)
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.linear(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            encoder_output=encoder_output.unsqueeze(0)
            #print ("encoder_outputs.shape "+str(encoder_output.shape))
            #print ("concat.shape"+str(torch.cat([hidden, encoder_output], 1).shape))
            energy = self.linear(torch.cat([hidden, encoder_output], 1))
            #print("energy.shape"+str(torch.transpose(energy,0,1).shape))
            # energy = self.other.dot(energy.view(-1))
            energy=torch.mm(self.parameter,torch.transpose(energy,0,1))
            #print("shape of final energy"+str(energy.shape))
            return energy

class FND(nn.Module):
    def __init__(self,method,attn_hidden_size,claim_feature_size,article_term_dim,claim_source_dim,article_source_dim,hidden_size,output_size):
        super(FND,self).__init__()
        self.output_size=output_size
        self.attn=Attn(method,claim_feature_size,article_term_dim,attn_hidden_size)
        self.dense_1=nn.Linear(article_term_dim+claim_source_dim+article_source_dim,hidden_size)
        self.dense_2=nn.Linear(hidden_size,hidden_size)
        self.pred = nn.Linear(hidden_size,output_size)
        
    def forward(self,claim_input,article_inputs,claim_source,article_source):
        attn_wts=self.attn(claim_input,article_inputs)
        #print("attention weights"+str(attn_wts))
        article_inputs=article_inputs.cuda()
        claim_source=torch.cuda.FloatTensor(claim_source).unsqueeze(0)
        article_source=torch.cuda.FloatTensor(article_source).unsqueeze(0)
        #print("FND:article_inputs"+str(article_inputs.shape))
        #print("FND:attn_wts shape"+str(attn_wts.shape))
        claim_specific_article=attn_wts.mm(article_inputs)
        #print("FND:claim_specific_article"+str(claim_specific_article.shape))
        concatenated_feature = torch.cat((claim_specific_article,claim_source,article_source),1)
        #print("shape of concatenated_feature"+str(concatenated_feature.shape))
        concatenated_feature=concatenated_feature.cpu()
        h1 = self.dense_1(concatenated_feature)
       
        h1 = F.relu(h1)
        # print("h1.shape"+str(h1.shape))
        h2 = self.dense_2(h1)
        h2 = F.relu(h2)
        pred = self.pred(h2)
        pred = 5*F.sigmoid(pred)
        return pred
print("start")
batch_generator = TrainTestSplit.getBatch('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/claimCredibility.csv','/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/claimReviews.csv',2)
X,Y = next(batch_generator)
print("got embeddings")
ATTN_HIDDEN_SIZE = 100
CLAIM_FEATURE_SIZE = len(X[0]['claim'])
ARTICLE_TERM_DIM = len(X[0]['article'][0])
CLAIM_SOURCE_DIM = len(X[0]['claimSource'])
ARTICLE_SOURCE_DIM = len(X[0]['articleSource'])
DENSE_LAYER_DIM = 100
OUTPUT_DIM = 1  #Regression
model = FND('concat',ATTN_HIDDEN_SIZE,CLAIM_FEATURE_SIZE,ARTICLE_TERM_DIM,CLAIM_SOURCE_DIM,ARTICLE_SOURCE_DIM,DENSE_LAYER_DIM,OUTPUT_DIM)
print(model)
print("model initalized")
model.cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.0001)
loss_fn = torch.nn.MSELoss() 
pred_y=[]
for i in range(len(Y)):
    print("batch\t"+str(i))
    x=torch.cuda.FloatTensor(X[i])
    y=torch.cuda.FloatTensor(Y[i])
    claim_input = x['claim']
    article_inputs = x['article']
    claim_source = x['claimSource']
    article_source = x['articleSource']
    claim_source=torch.cuda.FloatTensor(claim_source)
    article_source=torch.cuda.FloatTensor(article_source)
    optimizer.zero_grad()
    output = model(claim_input,article_inputs,claim_source,article_source)
    # print("output shape"+str(output.shape))
    pred_y.append(output)
    loss = loss_fn(output[0], y)
    print('Loss:'+str(loss))
    loss.backward()
    optimizer.step()
# test_batch_generator = TrainTestSplit.getBatch('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/claimCredibility.csv','/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/claimReviews.csv',20)
# X,Y = next(batch_generator)