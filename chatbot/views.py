from django.shortcuts import render
from django.http import HttpResponse

import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras
import numpy
import json
import random 

with open('chatbot/data.json') as file:
    data = json.load(file)

stemmer = LancasterStemmer()

model = keras.models.load_model('chatbot/model.h5')

words = ["'m",
 ',',
 'a',
 'admit',
 'afternoon',
 'any',
 'apply',
 'apprecy',
 'ar',
 'attend',
 'be',
 'bye',
 'can',
 'car',
 'confus',
 'cost',
 'criter',
 'dat',
 'deadlin',
 'do',
 'doe',
 'fee',
 'for',
 'funny',
 'good',
 'goodby',
 'gre',
 'hear',
 'hello',
 'hey',
 'hi',
 'how',
 'i',
 'intern',
 'is',
 'it',
 'jok',
 'know',
 'last',
 'lat',
 'me',
 'mean',
 'morn',
 'much',
 'my',
 "n't",
 'nee',
 'of',
 'pleas',
 'rep',
 'requir',
 'see',
 'stud',
 'submit',
 'tak',
 'tel',
 'thank',
 'that',
 'the',
 'thi',
 'to',
 'tuit',
 'understand',
 'univers',
 'want',
 'what',
 'when',
 'you']

labels = ['admission_requirements',
 'application_deadline',
 'goodbye',
 'greeting',
 'joke',
 'thank_you',
 'tuition_fee',
 'unknown']

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def home(request):

    if request.method == 'POST':
        input_text = request.POST['input_text']

        input_data = numpy.array([bag_of_words(input_text, words)])  # Convert to numpy array
        input_data = numpy.reshape(input_data, (input_data.shape[0], -1))  # Reshape to 2D array
        results = model.predict(input_data)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = random.choice(responses)
        return render(request, 'chatbot/home.html', {'response': response})
    else:
        return render(request, 'chatbot/home.html')