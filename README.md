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

FILES:==>

1.ex2NLP.py - the main proggram which contains the functions for calculating error rates for 
baseline case, BIGRAM_HMM, ADD_SMOOTH, PSEUDOWORDS, PSEUDOWORDS AND SMOOTH.
the main function runTests(.) runs the relevant functions for each algorithm and print the error rates to screen

2.ex2NLP.pdf - answer for Q1

3. confMatrix.txt - confusion Matrix for Pseudo Words and Smooth cases (VITERBI)

error rates - in detail -->

b) baseline case ->

------------> logLiklihood model for POS<----------

-errorRateTest - Known :  0.08283556857883467
-errorRateTest - UnKnown :  0.7895196506550218
-errorRateTest - total :  0.16360551008185265

c)

------------>HMM bigram model for POS<-------------

-errorRateTest - Known :  0.8677280636683417
-errorRateTest - UnKnown :  0.3894051835610276
-errorRateTest - total :  0.7821537312552149

--------------------------------------------

d)

------------>HMM Add-one smoothing for POS<-------------
-errorRateTest - Known :  0.867842390137348
-errorRateTest - UnKnown :  0.3894051835610276
-errorRateTest - total :  0.7822099520906566

--------------------------------------------
e_i)I used Pseudo_Words regex  dictionary for grouping common patterns in corpus -->

PSEUDOWORDSDICT = { "\\d+.?\\d+" : 'NUMBER', "^\\d+\\/\\d+\\/\\d+$" : 'DATE', "^\\d+.?\\d+\\$$" : 'PRICE' ,
                "^[A-Z]+$" : 'CAPITALS', "^[A-Za-z][.][A-Za-z]([.][A-Za-z])*$": 'NOTRICON', 
                "^[A-Z]+[a-z]*[.]$" : 'capPeriod',"^[A-Z]+[a-z]+$" :'NAMEOrFitstLetter',"^[A-Za-z]+[e][d]$" : 'ed_suffix',
                "^[A-Za-z]+[-][A-Za-z]+$" : 'multiple_words', "^[A-Za-z]+[t][i][o][n]$": "tionSuffix","^[A-Za-z]+[e][r]$": 'erSuffix',
                "^[A-Za-z]+[i][n][g]$": 'ingSuffix', "^[A-Za-z]+[s]$": 'plural' , "^[A-Za-z]+[l][y]$": 'lySuffix'}

e_ii)

------------>HMM Pseudo-Words for POS<-------------
-errorRateTest - Known :  0.5539556896993079
-errorRateTest - UnKnown :  0.4344431482418494
-errorRateTest - total :  0.5264355927843569

--------------------------------------------

e_iii)

------------>HMM Pseudo-Words and SMOOTHING for POS<-------------

-errorRateTest - Known :  0.5582273612098992
-errorRateTest - UnKnown :  0.43155714535584655
-errorRateTest - total :  0.5295999546392864

--------------------------------------------

conclusion-->

the baseline case performs much better error rates than any version of Viterbi algorithm.
I assume that the problem in this excersise, given the spesific corpus, is relatively simple,
and thus the baseline case achieve lower error rates than Viterbi. on more complex cases, the 
Viterbi should achieve significantly lower error rates.

I also can notice that given the groups created by Pseudo Words dictionary (see e_i), the Viterbi achieve
much better results, compared with the regular and smoothed bigram case.
although I assume creating more groups would decrease the error rate .

notice that smoothing alone without Pseudo Words does not achieve lower results than the regular bigram case,
beacuse smoothing dous not improve the probabilty of unknown words, but only the known words, so in total it
has small infuence.

from the confusion matrix (see confMatrix.txt) I can deduce that Viterbi algorithm with Pseudo Words and Smooth achieve the most
common errors on tags NN ,IN, AT.
