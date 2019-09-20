"""
Reza Shisheie
2708062
Lab 2: HTML data mining
"""

import numpy as np
import requests
from bs4 import BeautifulSoup
import cgi
import re
import urllib
import nltk
from tabulate import tabulate
from prettytable import PrettyTable

def fileSave(url):
    f = open('Doc1.txt','w')
    html = urllib.urlopen(url[0]).read()
    soup = BeautifulSoup(html,"lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    text = text.lower()    
    text = re.sub(' +',' ',text)
    text = re.sub('\n+',' ',text)
    f.write(text.encode("utf-8"))
    f.close()

    f = open('Doc2.txt','w')
    html = urllib.urlopen(url[1]).read()
    soup = BeautifulSoup(html,"lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    text = text.lower()    
    text = re.sub(' +',' ',text)
    text = re.sub('\n+',' ',text)
    f.write(text.encode("utf-8"))
    f.close()

    f = open('Doc3.txt','w')
    html = urllib.urlopen(url[2]).read()
    soup = BeautifulSoup(html,"lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    text = text.lower()    
    text = re.sub(' +',' ',text)
    text = re.sub('\n+',' ',text)
    f.write(text.encode("utf-8"))
    f.close()

    f = open('Doc4.txt','w')
    html = urllib.urlopen(url[3]).read()
    soup = BeautifulSoup(html,"lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    text = text.lower()    
    text = re.sub(' +',' ',text)
    text = re.sub('\n+',' ',text)
    f.write(text.encode("utf-8"))
    f.close()

    f = open('Doc5.txt','w')
    html = urllib.urlopen(url[4]).read()
    soup = BeautifulSoup(html,"lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    text = text.lower()    
    text = re.sub(' +',' ',text)
    text = re.sub('\n+',' ',text)
    f.write(text.encode("utf-8"))
    f.close()
    

    
 
def count_words(array, url, the_word, jj):

    wordLen = len(the_word)

    html = urllib.urlopen(url[jj]).read()
    soup = BeautifulSoup(html,"lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
      
    text = soup.get_text()
    text = text.lower()
    text = re.sub(' +',' ',text)
    text = re.sub('\n+',' ',text)
    
    for i in range(0,wordLen):
        result = 0
        tempResult = 0

        for j in range(0,len(the_word[i])):
            #print the_word[i][j]
            tempResult = re.findall('\\b'+the_word[i][j]+'\\b', text, flags=re.IGNORECASE)
            result = result + len(tempResult)
        
        array[jj][i] = result

    #print url[jj]
    #print array

    return array


#def cosine(array):
#    for i in range(0,len(array[0])):
#        for j in range(0,len(array[0])):
                
def cleanDup(array, col, des):
    lenCol = len(col)
    for i in range (0,len(array)):
        for j in range (0,len(array[0])):
            if j in col:
                array[i][j] = array[i][j] - array[i][des]
    return array
                
def printArray (array,word):
    t = PrettyTable([' ', word[0][0], word[1][0],word[2][0],word[3][0],word[4][0],word[5][0]])
    t.add_row(['Doc1', array[0][0], array[0][1], array[0][2], array[0][3], array[0][4], array[0][5]])
    t.add_row(['Doc2', array[1][0], array[1][1], array[1][2], array[1][3], array[1][4], array[1][5]]), 
    t.add_row(['Doc3', array[2][0], array[2][1], array[2][2], array[2][3], array[2][4], array[2][5]]), 
    t.add_row(['Doc4', array[3][0], array[3][1], array[3][2], array[3][3], array[3][4], array[3][5]]), 
    t.add_row(['Doc5', array[4][0], array[4][1], array[4][2], array[4][3], array[4][4], array[4][5]]), 

    print t

    
 
def main():
    print "started..."
    print ""
    
    url = ['https://www.csuohio.edu/engineering/eecs/faculty-staff',
    	   'http://engineering.case.edu/eecs/',
    	   'https://my.clevelandclinic.org/research',
    	   'https://en.wikipedia.org/wiki/Data_mining',
    	   'https://en.wikipedia.org/wiki/Data_mining']
    	   
#    word = ['Engineering','Professor', 'Research', 'Data', 'Mining', 'Data Mining']
    word = [['engineering', 'engineer', 'engineers'],
            ['professor', 'professors', 'prof'], 
            ['research', 'researcher', 'researchers', 'researching'], 
            ['data'], 
            ['mining'], 
            ['data mining']]

 
    array = np.zeros((len(url), len(word)))


    '''

    print tabulate(
    		   ['Doc1', array[0][0], array[0][2], array[0][3], array[0][4], array[0][5]], 
    		   ['Doc2', array[1][0], array[1][2], array[1][3], array[1][4], array[1][5]], 
    		   ['Doc3', array[2][0], array[2][2], array[2][3], array[2][4], array[2][5]], 
    		   ['Doc4', array[3][0], array[3][2], array[3][3], array[3][4], array[3][5]], 
    		   ['Doc5', array[4][0], array[4][2], array[4][3], array[4][4], array[4][5]], 
    		   headers=[' ', word[0][0], word[1][0],word[2][0],word[3][0],word[4][0],word[5][0]])
    '''
    #raw_input()

    
    fileSave(url)
    print "files saved..."
    print ""
    
    for j in range(0,len(url)):
        array = count_words(array, url, word, j)
    #print array
    printArray (array,word)
    print "array saved..."    
    print ""        
            
    array = cleanDup(array,[3,4], 5)
    #print array    
    printArray (array,word)
    print "array cleaned..."

            
 
if __name__ == '__main__':
    main()


'''
[[   3.    0.    5.    1.    0.    0.]
 [   6.    2.    3.    1.    0.    0.]
 [   0.    0.   14.    0.    0.    0.]
 [   6.    0.   23.  305.  181.  150.]
 [   0.    0.    0.    0.    0.    0.]]

[[   3.    0.    3.    0.    0.    0.]
 [   3.    2.    2.    0.    0.    0.]
 [   0.    0.    7.    0.    0.    0.]
 [   3.    0.   14.  214.  110.   79.]
 [   0.    0.    0.    0.    0.    0.]]


[[   3.    0.    4.    0.    0.    0.]
 [   5.    2.    3.    1.    0.    0.]
 [   0.    0.   11.    0.    0.    0.]
 [   6.    0.   23.  273.  178.  148.]
 [   0.    0.    0.    0.    0.    0.]]


'''

