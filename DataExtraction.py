# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:37:11 2018

@author: ayush
"""

import pandas as pd
import csv

#user_info=pd.read_csv('./NewsTrustData/NewsArticles.tsv',delimiter='\t',encoding='latin-1')
#print(list(user_info.columns.values)) #file header
#print(user_info.tail(2)) #last N rows



        
def getClaims(fname):
    with open(fname,'rb') as f:
        content = f.readlines()
    with open('claims.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)   
        wr.writerow(['source','text'])
        for x in content:
            #x = x.rstrip()
            line = x.split(b'\t')
            source = line[0]  #.strip('b\'').rstrip('\'')
            text = line[1]  #.strip('b\'').rstrip('\'')
            wr.writerow([source,text])
    myfile.close()



def getClaimsReviewsAndCredibility(fname):
    with open(fname,'rb') as f:
        content = f.readlines()
    with open('claimCredibility.csv','w',newline='') as credfile:
        with open('claimReviews.csv', 'w',newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)   
            wrcred = csv.writer(credfile, quoting=csv.QUOTE_ALL)
            wrcred.writerow(['claimId','claimLink','credibility'])
            wr.writerow(['claimId','Review','Reviewer'])
            ClaimId=0
            Reviews=[]
            Reviewers=[]
            credibility=0
            articleCount=0# to determine corrupted stories having more than one unrelated claims
            claimLink='' #claim source i.e. news article link
            for x in content:                     
                line = x.split(b'\t')
                if(line[0]==b'story-id'):
                    if(ClaimId):
                        if(articleCount == 1): #don't consider the story if having more than 1 news article
                            for i in range(len(Reviews)):                                
                                wr.writerow([ClaimId.rstrip(b'\r\n'),Reviews[i].rstrip(b'\r\n'),Reviewers[i]])
                            wrcred.writerow([ClaimId.rstrip(b'\r\n'),claimLink.rstrip(b'\r\n'),credibility])
                        Reviews = []
                        Reviewers = []
                        articleCount = 0
                    ClaimId = line[1]
                if(line[0]==b'newsarticle-link'):
                    if(articleCount==0):
                        claimLink = line[1]
                    articleCount = articleCount + 1
                if(line[0]==b'overall-ratingLabels'):
                    labels = line[1].split(b',')
                    if(b' Credibility' in labels):
                        credIndex = labels.index(b' Credibility')
                    else:
                        credIndex = -1
                        credibility = 0
                if(line[0]==b'overall-ratings')   :
                    if(credIndex>=0):
                        credibility = line[1].split(b',')[credIndex]
                if(line[0]==b'member-review'):
                    review = line[1].split(b"by")[1]
                    name= review.split(b'-',2)[0]
                    #remove dates and name and loginto comment from reviews  #data cleaning
                    #review = review.replace()
                    Reviewers.append(name)
                    Reviews.append(review)                   

def normalizeSourceAttributes():
    claimSourceFrame = pd.read_csv('claim_sources.csv', names=['source','a1','a2','a3','a4','a5','a6','a7'])
    claimSourceNorm = claimSourceFrame.loc[:,['a1','a2','a3','a4','a5','a6','a7']]
    #print(claimSourceFrame)
    claimSourceNorm = (claimSourceNorm-claimSourceNorm.min())/(claimSourceNorm.max()-claimSourceNorm.min())
    claimSourceNorm.insert(loc=0, column='source', value=claimSourceFrame['source'])
    #claimSourceNorm['source']=claimSourceFrame['source']
    claimSourceNorm.to_csv('claim_sources.csv',header=False,index=False)
    
    articleSourceFrame = pd.read_csv('members.csv',names=['name','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'])
    articleSourceNorm = articleSourceFrame.loc[:,['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12']]
    articleSourceNorm = (articleSourceNorm-articleSourceNorm.min())/(articleSourceNorm.max()-articleSourceNorm.min())
    articleSourceNorm.insert(loc=0, column='name', value=articleSourceFrame['name'])
    articleSourceNorm.to_csv('members.csv',header=False,index=False)
getClaims('./NewsTrustData/NewsArticles.tsv')
getClaimsReviewsAndCredibility('./NewsTrustData/NewsTrustStories.tsv')
normalizeSourceAttributes()
