import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.classify import NaiveBayesClassifier, accuracy
import csv,string,random,pickle

#Naïve Bayes

def naiveBayes():
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

  words = word_tokenize(positive) + word_tokenize(negative)

  #preprocess

  ##remove unnecessary words
  words = [w for w in words if w.lower() not in eng_stopwords]
  
  ##remove punctuation
  words = [w for w in words if w.lower() not in string.punctuation]
  
  ##remove not words
  words = [w for w in words if w.isalpha()]


  #POS tagging and NER
  tag = pos_tag(words)
  ner = ne_chunk(tag)
  

  stemmer = PorterStemmer()
  words = [stemmer.stem(word) for word in words]

  freqD = FreqDist(words)
  words = [word for word in freqD.most_common(1000)]

  list_labeled_sentence = []
  for sentence in positive.split("\n"):
      list_labeled_sentence.append((sentence, "positive"))
      
  for sentence in negative.split("\n"):
      list_labeled_sentence.append((sentence, "negative"))


  #extract data features
  datasets = []
  for sentence, label in list_labeled_sentence:
    dict = {}
    wordz = word_tokenize(sentence)

    for w in words:
      dict[w] = w in wordz

    datasets.append((dict, label))

  random.shuffle(datasets)
  mid_point = int(len(datasets) * 0.5)
  training = datasets[:mid_point]
  testing = datasets[mid_point:]
  
  classifier = NaiveBayesClassifier.train((training))

  print(f"Created data with accuracy : {accuracy(classifier, testing) * 100 : .2f}% accuracy")
  file = open("data.pickle", "wb")
  pickle.dump(classifier, file)
  file.close()


def menu():
  naiveBayes()
  file = open("data.pickle", "rb")
  classifier = pickle.load(file)
  file.close()
  while True:
    print("Cyberbullying Tweets Identification\n")
    print("1. Check Cyberbully or not")
    print('2. View Model POS Tag')
    print('3. View Model NER')
    print("4. Exit")
    ch = int(input(">> : "))
    if ch == 1:
      print('input menu')
    elif ch == 2:
       print('view pos')
    elif ch == 3:
       print('view ner')
    elif ch == 4:
      exit(0)




menu()


