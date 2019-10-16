# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:54:23 2018

@author: Ayush jain
"""
import pandas as pd
from urllib.parse import urlparse
import Preprocess
#import tldextract
print("traintestsplit.py")
def ClaimsTrainTestSplit():
    claimFrame = pd.read_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/claimCredibility.csv')
    reviewsFrame = pd.read_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/claimReviews.csv')
    claimTrainFrame = claimFrame.iloc[:40000][:]
    claimTestFrame = claimFrame.iloc[40000:][:]
    reviewsTrainFrame = reviewsFrame.loc[reviewsFrame['claimId'].isin(claimTrainFrame['claimId'])]
    reviewsTestFrame = reviewsFrame.loc[reviewsFrame['claimId'].isin(claimTestFrame['claimId'])]
    claimTrainFrame.to_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/claimCredibility.csv')
    claimTestFrame.to_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/claimCredibility.csv')
    reviewsTrainFrame.to_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/claimReviews.csv')
    reviewsTestFrame.to_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/claimReviews.csv')
    
#ClaimsTrainTestSplit() 

def getClaimTextFromLink(claimLink):
    #if(not isinstance(claimLink,str)):
    #    claimLink = claimLink.iloc[0]
    claims = pd.read_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/claims.csv',index_col=0)
    return claims.loc[claimLink,:].tolist()[0]

def getReviewsFromClaimId(claimId,reviewFile):
    reviews = pd.read_csv(reviewFile)
    reviews = reviews.loc[reviews['claimId'].isin([claimId])]
    #print(reviews)
    reviews = reviews.loc[:,['Review','Reviewer']]
    reviewlist = []
    reviewerslist = []
    for i in reviews.index:
        
        review = reviews.loc[i,'Review']
        reviewer = reviews.loc[i,'Reviewer']
        reviewlist.append(review)
        reviewerslist.append(reviewer)
    return reviewlist,reviewerslist

def getClaimSourceAttrFromClaimLink(claimLink):
    #if( not isinstance(claimLink,str)):
    #    claimLink = claimLink.iloc[0]
    domain = urlparse(claimLink).netloc
    #print(domain)
    #domain = tldextract.extract(claimLink).domain
    sourceFrame = pd.read_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/claim_sources.csv',names=['source', 'a1','a2','a3','a4','a5','a6','a7'], header=None)
    sourceFrame = sourceFrame.loc[sourceFrame['source'].str.contains(domain)]    
    sourceFrame = sourceFrame.drop_duplicates(subset=['source'], keep='last') #removing duplicate rows
    return sourceFrame.loc[:,['a1','a2','a3','a4','a5','a6','a7']].values
    
def getReviewerAttrFromName(name)  :   
    name = name.strip(' ').lower()
    name = name.replace(' ','-')
    reviewer = pd.read_csv('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/members.csv',index_col=0,names=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'])
    reviewer = reviewer[~reviewer.index.duplicated(keep='last')] #removing duplicate rows
    return reviewer.loc[name,:].values

def getBatch(claimFile,reviewFile,batchSize):
    claimFrame = pd.read_csv(claimFile,index_col=0)  
    claimFrame = claimFrame[~claimFrame.index.duplicated(keep='first')] #removing duplicate rows
    X =[]
    Y = []
    counter=0
    while True:               
        for claimId in claimFrame.index.tolist():
            # print('caught!\n')
            try:
                reviews,reviewers = getReviewsFromClaimId(claimId,reviewFile)            
                claimSourceEmbedding = getClaimSourceAttrFromClaimLink(claimFrame.loc[claimId,'claimLink'])
                credibility = claimFrame.loc[claimId,'credibility']
            #if(isinstance(credibility,pd.Series)): #above statement may return a series if duplicate entries are there in claimframe
            #    credibility = credibility.iloc[0]            
                claimText = getClaimTextFromLink(claimFrame.loc[claimId,'claimLink'])
            except:            
                continue           
            claimEmbeddings = Preprocess.getUniversalSentenceEncoding(claimText)
            print("no. of reviews"+str(len(reviews)))
            #print(reviews)
            if len(reviews) > 0:
                for i in range(len(reviews)):
                    tokens = Preprocess.tokenizer(reviews[i])
                    #articleTerms = tokens             
                    articleTermEmbeddings = Preprocess.getContextualisedWordEmbeddings(tokens)
                    reviewerEmbedding = getReviewerAttrFromName(reviewers[i])
                    x = {'claimId':claimId,'claim' : claimEmbeddings, 'article': articleTermEmbeddings,'claimSource':claimSourceEmbedding,'articleSource':reviewerEmbedding }
                    y = credibility
                    X.append(x)
                    Y.append(y)
                    counter = counter +1
                    if(counter == batchSize):
                       yield X,Y
                       X=[]
                       Y=[]
                       counter = 0
            
    #while True:
        
#getClaimTextFromLink("b'http://www.npr.org/templates/story/story.php?storyId=121529261'")

#TrainDATa
#XTrain,YTrain = next(getBatch('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/claimCredibility.csv','/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/claimReviews.csv',3000))
#print(X)
#print(Y)
#ClaimsTrainTestSplit()

#with open('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/XTrain.pickle', 'wb') as handle:
#    pickle.dump(XTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Train/YTrain.pickle', 'wb') as handle:
#	pickle.dump(YTrain,handle, protocol=pickle.HIGHEST_PROTOCOL)

##Test Data## 
#XTest,YTest = next(getBatch('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/claimCredibility.csv','/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/claimReviews.csv',1000))

#print(X)
#print(Y)
#ClaimsTrainTestSplit()

#with open('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/XTest.pickle', 'wb') as handle:
#    pickle.dump(XTest, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('/media/sdb/sanjay/IR/debunking-fake-news/newstrust_final/Test/YTest.pickle', 'wb') as handle:
#        pickle.dump(YTest,handle, protocol=pickle.HIGHEST_PROTOCOL)

