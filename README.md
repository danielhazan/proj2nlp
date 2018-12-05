# proj2nlp

programm written for project 2 in NLP course (3rd year) in huji which compute error rates of some 
learning algorithms for POS tagging. the train corpus taken from nltk.brown library module.
the first algorithm used for POS tagging is the baseline case using log-likelihood estimator on train corpus. 
the second tagger is Hidden Markov Model which uses the dynamic Viterbi algorithm using emmision and transition
probabilties to estimate the sequence of tagging in the coprpus sentences. 
the last two algorithms add improvement on the previous HMM bigram model for tagging: using add-1 smoothing for
known words in the corpus train, and PSEUDO-WORDS which groups the train vocabulry into patterns (using regex) 
which will improve the estimation of POS tags in the test corpus. in the README I supply the results of error rates 
using each of the above  models and conclude with the confusion Matrix which indicates the most common taggs that 
achieved most error rates
