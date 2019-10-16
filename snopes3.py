import os
import json
import pandas as pd
from bs4 import BeautifulSoup as bs
import urllib3 
import re
import sys
from nltk.tokenize import word_tokenize
import nltk, string, numpy,math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
nltk.download('punkt')
nltk.download('wordnet')
stemmer = nltk.stem.porter.PorterStemmer()


def StemTokens(tokens):
     return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def StemNormalize(text):
     return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
def idf(n,df):
    result = math.log((n+1.0)/(df+1.0)) + 1
    return result
def visible(element):
            if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
                return False
            elif re.match('<!--.*-->', str(element.encode('utf-8'))):
                return False
            return True
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

def similarity(claim,article_snippets):
    docs=[]
    docs.append(claim)
    for i in range(0,len(article_snippets)):
        docs.append(article_snippets[i])
    LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
    LemVectorizer.fit_transform(docs)
    tf_matrix = LemVectorizer.transform(docs).toarray()
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    def cos_similarity(textlist):
        tfidf = TfidfVec.fit_transform(textlist)
        return (tfidf * tfidf.T).toarray()
    cos_sim_mat=cos_similarity(docs)
    a= list(cos_sim_mat[0])
    k=second_largest(a)
    l=a.index(k)
    # print ("k_index \t"+str(k))
    # print(docs[l])
    return docs[l] 
def crawl(df):
    final_article=[]
    final_df=pd.DataFrame(columns=('claim_id','claim_file','claim','article_link','article','article_source','credibility'))
    k=0
    for i,row in df.iterrows() :
        article= row["article_link"]
        claim=row["claim"]
        file=row["claim_file"]
        http = urllib3.PoolManager()
        url=article
        print (file)
        try:
            response = http.request('GET', url)
        except:
            print("the row \t "+str(i)+"\t has been removed")
            df.drop(i)
            continue    
        print("crawl_index \t"+str(i))
        # print(article)
        soup=bs(response.data,'html.parser')
        data = soup.findAll(text=True)
        result = filter(visible, data)
        text=list(result)
       
        text=[e for e in text if e not in ('\n','👤',' ')]
        # print(text)
        if len(text) > 10:
            # df.drop(i)
            # continue
            # print(text)
            final_text=[]
            for j in text:
                if len(j) > 100 :
                    # print("*****************")
                    # print(len(j))
                    # print("----------")
                    # # print (i)
                    # print("-----------------------------------------------")
                    final_text.append(j)
                #print(i)
                #print(final_text[i])
            if len(final_text)>2 :
                article=similarity(claim,final_text) #toDO
                final_df.loc[k]=[row['claim_id'],row['claim_file'],row['claim'],row['article_link'],article,row['article_source'],row['credibility']]
                k=k+1
                final_article.append(article) 
           
    return final_df
def create_df(json_files):
    df=pd.DataFrame(columns=('claim_id','claim_file','claim','article_link','article_source','credibility'))
    j=0
    k=0
    for file in json_files: 
        j=j+1
        file_path=path+'/'+file
        with open(file_path,'r') as f:
            file_data=json.load(f)
        if len(file_data["Google Results"][0]["results"]) > 7 :
            claim=file_data["Claim"] 
            cred=file_data["Credibility"]
            for i in range(0,7):
                article=file_data["Google Results"][0]["results"][i]['link']
                article_source=file_data["Google Results"][0]["results"][i]["domain"]
                df.loc[k]=[k,file,claim,article,article_source,cred]
                k=k+1
    df.to_csv('snopes_refined.csv', sep='\t')
def main(a,b,filename):
    path="/media/sdb/sanjay/IR/debunking-fake-news/Snopes/Snopes"
    if os.path.exists(path):
        json_files=os.listdir(path)
    #create_df(json_files)
    df=pd.read_csv('snopes_refined/snopes_refined.csv',sep='\t')
    df_final=crawl(df[int(a):int(b)])
    df_final.to_csv('snopes_final/'+filename,sep='\t')
         


if __name__== "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])

