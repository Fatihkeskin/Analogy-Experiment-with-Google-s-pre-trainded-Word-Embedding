Embeddings are generally learned on a large corpus. Once the embeddings are trained, how can we measure the quality
of learned embeddings?
- Actually the concept of quality is difficult to measure objectively because vector representation is not independent
of the learning algorithm in terms of how good it is. but several methods can be used to test the accuracy and quality
of program. Shortly a word representation mainly depends on the dataset and the learning algorithms. One of the
methods to evaluate an embedding’s quality is the analogy. Analogy is question to find the relationships between words.
For example, "Malatya" is to "Kayısı", what "Kocaeli" is to... Here, the answer is "Pi¸smaniye". The other method
to measure the quality of the word vectors is a human survey. We can ask enough humans to score the relationship
between words 1 to 10. Then these results can be compared with the trained model.

In cosine similarity part, all we have to do is to upload the famous Google 300 dimensional word vectors to our program. Then
search the corresponding vectors for the words in questions. Then doing algebraic operations on them to obtain a result
vector. 


In Document Classification task, i declared document representations as vectors based on doc2vec model to classify them. We
know that the statistical probabilities of documents can be very large and sparse using bag-of-words like algorithms.
The need of dense and meaningful representation for documents is clear. Thanks to Mikolov, now we can vary the
word2vec concept as doc2vec. Now, we can have vectors for documents, and we can classify them. For this task, the
movie id’s, movie plots, and movie genres are provided in repo. I used the first 2000 lines to learn, the remaining lines are
for the testing. So it is needed to read the training lines, remove stopwords to reduce irrevelant data ratio, tag the movie
plots, create a doc2vec model, build a vocabulary, train the vocabulary, apply logistic regression classifier and evaluate
the results.
