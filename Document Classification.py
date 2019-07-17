import time
import logging
import csv
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)                             #a nice method to see the progress not to get bored

""" Task 02: Document Classification """
print("TASK 02: Document Classification")

start_time = time.time()                                                                                                #measuring program efficiency

stopWords = set(stopwords.words('english'))                                                                             #miracle method from nltk to reduce irrevelant data

tags_index = {'sci-fi': 1 , 'action': 2, 'comedy': 3, 'fantasy': 4, 'animation': 5, 'romance': 6}                       #doing enumeration to ease supervised training


#Reading the given data
rawData = []
testData = []
lineCounter = 0
with open('tagged_plots_movielens.csv', 'r') as movies:
    movieCsv = csv.reader(movies, delimiter=',', quotechar='"')
    for lines in movieCsv:
        if (lineCounter == 0):                                                                                          #skip first line
            lineCounter += 1
            continue
        if (lineCounter < 2001):                                                                                        #read training data
            words1 = []
            for word in lines[2].split():
                if word not in stopWords:
                    words1.append(word.lower())
                if len(word) < 2:                                                                                       #discarding the same words
                    continue
            rawData.append(TaggedDocument(words=words1, tags=[tags_index.get(lines[3], 8)]))
        else:                                                                                                           #read test data
            words1 = []
            for word in lines[2].split():
                if word not in stopWords:
                    words1.append(word.lower())
                if len(word) < 2:                                                                                       #discarding the same words
                    continue
            testData.append(TaggedDocument(words=words1, tags=[tags_index.get(lines[3], 8)]))
        lineCounter += 1
#print(rawData)
#print(testData)

#create the training object
#parameters; algorithm is paragraph vector, dimension 300, window size 10, activation function= softmax, working with 4 threads, 10 negative sampling used, learning rate 0.025 to 0.001
trainingObject = Doc2Vec(dm=1, vector_size=300, dbow_words=1, negative=10, hs=1, min_count=2, sample = 0, workers=4, alpha=0.025, min_alpha=0.001)
#building the vocabulary from training data
trainingObject.build_vocab([x for x in rawData])
#print(trainingObject.build_vocab([x for x in rawData]))
#start training
trainingObject.train(rawData, total_examples=len(rawData), epochs=10)
#we need to save it
trainingObject.save('./trainedModel.d2v')
#print(trainingObject.save('./trainedModel.d2v'))


#creating the feature vectors from raw and test data
trainingMovieTags = []
trainingLogreg = []
comparingTestData= []
predictingLogreg= []
for itm in rawData:
    trainingMovieTags.append(itm.tags[0])
    trainingLogreg.append(trainingObject.infer_vector(itm.words, steps=20))

for tkn in testData:
    comparingTestData.append(tkn.tags[0])
    predictingLogreg.append(trainingObject.infer_vector(tkn.words, steps=20))

#logistic regression
regressionStep = LogisticRegression( C=1.0, solver='liblinear', multi_class='ovr', n_jobs=1)
regressionStep.fit(trainingLogreg, trainingMovieTags)
predictedNumpyArray = regressionStep.predict(predictingLogreg)


""" Evalutaion """
print("Calculating the Accuracy")
correctGuesses = 0
totalGuesses = 0
for a,b in zip(predictedNumpyArray, comparingTestData):
    if(a==b):
        correctGuesses += 1
    totalGuesses +=1

print('The Accuracy: %' , end='')
print(100*correctGuesses/totalGuesses)


print("\nEND OF THE ASSIGNMENT")

print("\n--- %s seconds ---" % (time.time() - start_time))
