
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') #for NER
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import json
import numpy as np
from keras.models import load_model  #USE JOBLIB instead
import random
import pickle

from googlesearch import search
import requests
from lxml import html
from bs4 import BeautifulSoup

import re
import calendar

from termcolor import colored #for colored text in response



#load pickled model
model = load_model("drive/My Drive/phoebe/chatbot_model.h5")
#load files
intents = json.loads(open('drive/My Drive/phoebe/intents.json').read())
words = pickle.load(open('drive/My Drive/phoebe/words.pkl','rb'))
classes = pickle.load(open('drive/My Drive/phoebe/classes.pkl','rb'))

#define our global variable
global context
context = {'userID': 'none'} #will discard later


#function
def bow(sentence, words, show_details=True): #show_details = false for deployment
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    #print(return_list)
    return return_list

#function to find location-entity in user query
def ner_location(query): #find POS of query words, cleaned sentence, no symbols
    tok = nltk.word_tokenize(query)
    res = nltk.pos_tag(tok)
    #chunking-> adding structure above and below the tagged sentences
    res_chunk = nltk.ne_chunk(res)
    result = []
    for x in str(res_chunk).split('\n'):
      if '/NN' in x:  
          names = re.findall(r'([a-zA-Z]+/NN[a-zA-Z])', x)
          for l in names:
            result.extend([l[0:-4]])
    
    return result

#print(ner_location("What is the weather in Bangalore and Assam?")) #gives all the locations - so lets scrap all

#google weather function
def search_weather(query):
  #find if location exist in query
  query = re.sub('[^a-zA-Z]', ' ', query) #remove symbols and digits
  location_list = list(ner_location(query))
  if len(location_list) == 0:
    location_list = []
    loc = input(colored("You need to specify which location!\n", "red"))
    print(colored('Checking...\n', "red"))
    location_list.extend([loc])
  for location in location_list:
      answer = "The weather in "+location+" is "
      try:
        article = ''
        search_result_list = list(search(query, lang='en', num=10, pause=1))
        page = requests.get(search_result_list[0]) #scraping weather.com
        
        if page.status_code == 200:
          soup = BeautifulSoup(page.text, features='lxml')
          
          #all_articles = soup.find('div', {"class" : "_-_-components-src-organism-CurrentConditions-CurrentConditions--primary--2DOqs" })
          all_articles = soup.find('span', {"class": "CurrentConditions--tempValue--3KcTQ"})
          if all_articles != None:
            result = all_articles.text  #only temperature
            ar = soup.find('div', {'data-testid': "precipPhrase"}).text  #details
            
            print(colored(answer+result+'F with '+str(ar), "red"))
          else: 
            raise colored(Exception("Sorry, facing an error"), "red")
      
      except Exception as e:
        print(str(e))
        result = "I can not find an answer for that. Try a different query!\n"
        print(colored(result, "red"))
  

#function to find name-entity in user query
def ner_find(query): #fins POS of query words
    tok = nltk.word_tokenize(query)
    res = nltk.pos_tag(tok)
    #chunking-> adding structure above and below the tagged sentences
    res_chunk = nltk.ne_chunk(res)
    result = []
    for x in str(res_chunk).split('\n'):
      if '/NNP' in x:  
          names = re.findall(r'([a-zA-Z]+/NN[a-zA-Z])', x)
          for l in names:
            result.extend(l[0:-4])
            result.extend(' ')

    return ''.join(result)

#google Celebrity AGE on wiki function
def search_wiki(query):
  #NER for celebrity name
  name = ner_find(query)
  #query = "How old is "+name
  answer = name+'is '
  try:
    article = ''
    search_result_list = list(search(query, lang='en', num=10, pause=1))
    page = requests.get(search_result_list[0]) #scraping wiki on the right card
    
    if page.status_code == 200: #all okay that is
      soup = BeautifulSoup(page.text, features='lxml')
      
      all_articles = soup.find('span', {"class": "noprint ForceAgeToShow"})
      if all_articles != None:
        result = all_articles.text  #only age number
        ar = soup.find('span', {'class': "bday"}).text  #birthday in date
        #convert birthdate to month, year format
        date = calendar.month_name[int(ar.split('-')[1])]
        
        print(colored(answer+str(re.findall(r'[0-9]+', result)[0])+' years old. Born on '+str(ar.split('-')[2])+' '+str(date)+" "+str(ar.split('-')[0]), "red"))
        return
      else: 
        raise colored(Exception("Sorry, facing an error!"), "red")
  
  except Exception as e:
    print(str(e))
    result = "I can not find an answer for that. Try a different query!\n"
    print(colored(result, "red"))
    return


#function that calls other google search functions
google_search_func = {'weather': search_weather, 'wiki': search_wiki} #argument is passed when called

#driver function version #2
def chatbot_response_2(msg, userID='userID', show_details = False):
    results = predict_class(msg, model)
    
    if results:
      for i in intents['intents']:
          # find a tag matching the first result
          if i['tag'] == results[0]['intent']:
                # set context for this intent if necessary
              if 'context' in i:
                  if show_details: print ('context:', i['context'])
                  context['userID'] = i['context']
                  # check if this intent is contextual and applies to this user's conversation
                      
              if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context['userID']):
                if show_details: print ('tag:', i['tag'])
                    # a random response from the intent
                response = random.choice(i['responses'])
                print(colored(response, "red"))
                if 'Checking' in response.split('.')[0]:
                  google_search_func[i['tag']](msg)  #when 'checking...'  in response we know to call this function to navigate further
              if i['tag'] in ['thanks', 'goodbye']:
                response = input(colored("Anything else?\n", "red"))
                if any( x in response.lower() for x in ["no", "nope", "nah", "no thanks"]):
                  return colored("Okay bye.", "red")

      results.pop(0)
    #else:
    #  if 'old' in msg:
    #    print(i['responses'][0])
    #    google_search_func[i['tag']](msg)  #call based on tag, print from function directly
           #have to respond and then call google search function acc to context
      print("\n")
      msg = input()
      return chatbot_response_2(msg)

#TEST--------------------------------------------------------------------------------------------

user_query = "Hi Phoebe"
print(chatbot_response_2(user_query))

#Retrain model to include new intents tag
#Next feature -> voice to voice (create recording)


