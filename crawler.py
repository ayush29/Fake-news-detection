from bs4 import BeautifulSoup as bs
import numpy as numpy
import urllib3 
import requests
# import nltk
import re
# http = urllib3.PoolManager()

url='http://www.snopes.com/what-does-koko-know-about-climate-change-nothing/'
# response = http.request('GET', url)
response=requests.get(url)
soup=bs(response.content,'html.parser')
data = soup.findAll(text=True)
 
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True
 
result = filter(visible, data)
text=list(result)
final_text=[]
for i in text:
    if len(i) > 100 :
        final_text.append(i)
for i in final_text:
    print(i)
    print("________________________|||||||||||||_______________________")
   
#print(final_text)        

    