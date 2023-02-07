import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import csv

#Naïve Bayes

def NaiveBayes():
    #load words
    
    positive,negative = '', ''
    with open('./Dataset/positive.csv', 'r',newline='',encoding="utf8") as csvfile:
      positiveCSV = list(csv.reader(csvfile))
      for i in positiveCSV:
         positive += str(i[0])
    
    with open('./Dataset/negative.csv', 'r',newline='',encoding="utf8") as csvfile:
      negativeCSV = list(csv.reader(csvfile))
      for i in negativeCSV:
         negative += str(i[0])
         
    eng_stopwords = stopwords.words('english')


    # print(positive)
    # words = word_tokenize(positive)
    words = word_tokenize(positive) + word_tokenize(negative)

    # print(words)
    #preprocess


    ##remove unnecessary
    words = [w for w in words if w.lower() not in eng_stopwords]
    
    # ##remove punctuation
    words = [w for w in words if w not in string.punctuation]
    
    # ##remove not words
    words = [w for w in words if w.isalpha()]
    
    

NaiveBayes()




