import time
import logging
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)                             #useful method i learned during development process

""" Task 01 Analogy Experiments
print("TASK 01: Loading corpus into the Main Memory") """

start_time = time.time()                                                                                                #start the timer to measure the execution time of the program

googleModel = KeyedVectors.load_word2vec_format(datapath("GoogleNews-vectors-negative300.bin"), binary=True, limit=100000)  #load the famous google dataset with 100000 words limit

correctGuessNumber = 0
guessNumber = 0

modelDict = dict({})                                                                                                    # we upload the model into the dictionary to make things faster
for idx, key in enumerate(googleModel.wv.vocab):
    modelDict[key] = googleModel.wv.get_vector(key)



    #print(modelDict.keys())

firstWord = ""
secondWord = ""
thirdWord = ""
targetWord = ""
lineCounter = 1
with open('word-test.v1.txt') as theQuestions:                                                                          #read the analogies
    for line in theQuestions:
        if( 12014 < lineCounter or lineCounter < 510 or line[0] ==':'): #12014 510                                      #narrow it down to 11504 analogies
            lineCounter+=1
            continue
        firstWord = line.split()[0]
        secondWord = line.split()[1]
        thirdWord = line.split()[2]
        targetWord = line.split()[3]
        """if (modelDict.get(thirdWord) = None or modelDict.get(firstWord) = None or modelDict.get(secondWord) = None):
            lineCounter += 1
            continue"""
        try:
            result = np.add(modelDict[thirdWord], np.subtract(modelDict[secondWord], modelDict[firstWord]))             #finding the relation vector
        except KeyError as e:                                                                                           #if given word does not exist in our dictionary
            print("The key has not found in limited 100000 words vocabulary continuing...")
            lineCounter+=1
            continue
        #print(result)
        modelAnswer = ""                                                                                                #program's prediction of the analogy
        highestSimilarity = 0
        for k,v in modelDict.items():
            currentSimilarity = np.dot(result, v) / (np.linalg.norm(result) * np.linalg.norm(v))                        #applying the cosine similarity
            if(currentSimilarity > highestSimilarity and k != firstWord and k != secondWord and k != thirdWord):        #increase the accuracy by discarding the words already in the analogy
                highestSimilarity = currentSimilarity                                                                   #new max
                modelAnswer = k
                #print(highestSimilarity)
                #print(w)
        if(modelAnswer == targetWord ):
            correctGuessNumber +=1
        guessNumber+=1
        print("Found the answer as:" , modelAnswer , "||| Current Accuracy:", correctGuessNumber*100/guessNumber)
        lineCounter+=1
#Main scope
#print(correctGuessNumber*100/guessNumber)

print("\nEND OF THE ASSIGNMENT")

print("\n--- %s seconds ---" % (time.time() - start_time))
