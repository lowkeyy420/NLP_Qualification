import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.classify import NaiveBayesClassifier, accuracy
import csv,string,random,pickle
from os import system , name
from urllib import request

eng_stopwords = stopwords.words('english')


def clear():
  if name == 'nt':
    _ = system('cls')
  else:
    _ = system('clear')

#Naïve Bayes
def naiveBayes():
  print("Loading Data...\nHint, positive -> not cyberbullying , negative -> cyberbullying")

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
        
  words = word_tokenize(positive) + word_tokenize(negative)

  #preprocess

  ##remove unnecessary words
  words = [w for w in words if w.lower() not in eng_stopwords]
  
  ##remove punctuation
  words = [w for w in words if w.lower() not in string.punctuation]
  
  ##remove not words
  words = [w for w in words if w.isalpha()]
  
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

  save_pickle(classifier)

  clear()
  print(f"Created data with accuracy : {accuracy(classifier, testing) * 100 : .2f}% accuracy")
  return classifier

def save_pickle(classifier):
  file = open("data.pickle", "wb")
  pickle.dump(classifier, file)
  file.close()

def load_pickle():
  try:
    file = open('data.pickle', 'rb')
    _pickle = pickle.load(file)
    file.close()
    return _pickle
  except FileNotFoundError:
    raise Exception("No file found")

def checkCyberBullying(classifier):
  text = input("Enter a text / tweet : ")
  words = word_tokenize(text)
  #POS tagging
  text = pos_tag(words)

  words = [w for w in words if w.lower() not in string.punctuation]
  words = [w for w in words if w.lower() not in eng_stopwords]
  words = [w for w in words if w.isalpha()]
  lemmatizer = WordNetLemmatizer()

  stemmed_words = [lemmatizer.lemmatize(word) for word in words]

  words = pos_tag(stemmed_words)
  result = classifier.classify(FreqDist(words))

  #NER
  ner = ne_chunk(text)
  return result,ner


def showCorpus():
  url = 'https://www.gutenberg.org/files/63919/63919.txt'
  res = request.urlopen(url).read().decode('utf-8')
  words = res.split()

  counter = 0
  for word in words:
    counter += 1
    synsets = wordnet.synsets(word)
    for synset in synsets:
      print(f"{synset} : {synset.definition()}")
      for lemma in synset.lemmas():
        print(f"synonim -> {lemma.name()}")
        for antonym in lemma.antonyms():
          print(f"anonym -> {antonym.name()}")
      print('\n')
    if counter >= 25:
      break

def menu():
  classifier = load_pickle()

  if(classifier is None):
    classifier = naiveBayes()
  
  while True:
    print("Cyberbullying Tweets Identification")
    print("1. Check Cyberbullying or not")
    print("2. Show Corpus")
    print("3. Exit")
    option = int(input(">> "))
    if option == 1:
      result = checkCyberBullying(classifier)
      clear()
      print(f"Result : {result[0]}")
      print(f"Structure : {result[1]}")
    elif option == 2:
      clear()
      showCorpus()
    elif option == 3:
      exit(0)




menu()

# nltk.download('stopwords')
# nltk.download('words')
# nltk.download('maxent_ne_chunker')
# nltk.download('wordnet')
# nltk.download('gutenberg')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
