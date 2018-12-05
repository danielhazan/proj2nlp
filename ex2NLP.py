import nltk
#nltk.download('brown')
from nltk.corpus import  brown
from collections import Counter, defaultdict
import re
from nltk import word_tokenize


BIGRAM_HMM = 0
ADD_SMOOTH = 1
PSEUDOWORDS = 2
MINFREQUENCY = 5
PSEUDOWORDSANDSMOOTH = 5

PSEUDOWORDSDICT = { "\\d+.?\\d+" : 'NUMBER', "^\\d+\\/\\d+\\/\\d+$" : 'DATE', "^\\d+.?\\d+\\$$" : 'PRICE' ,
                "^[A-Z]+$" : 'CAPITALS', "^[A-Za-z][.][A-Za-z]([.][A-Za-z])*$": 'NOTRICON', "^[A-Z]+[a-z]*[.]$" : 'capPeriod',
                "^[A-Z]+[a-z]+$" :'NAMEOrFitstLetter',"^[A-Za-z]+[e][d]$" : 'ed_suffix',"^[A-Za-z]+[-][A-Za-z]+$" : 'multiple_words',
                "^[A-Za-z]+[t][i][o][n]$": "tionSuffix","^[A-Za-z]+[e][r]$": 'erSuffix',"^[A-Za-z]+[i][n][g]$": 'ingSuffix',
                "^[A-Za-z]+[s]$": 'plural' , "^[A-Za-z]+[l][y]$": 'lySuffix'}################


df = brown.words()

newsPortionSents = brown.tagged_sents(categories = 'news')

trainSizeCorpus  = round(len(newsPortionSents) * 0.9)
trainCorpus = newsPortionSents[:trainSizeCorpus]
testCorpus = newsPortionSents[trainSizeCorpus:]


#2_b_ii

def findTagsOf(trainText, taggedText):
    """return a tag for each word in trainText"""

    wordsToTags = []

    for tuple in taggedText:
        for word in trainText:
            if word[0] == tuple[0]:
                wordsToTags.append((word[0],tuple[1]))
        continue

    return wordsToTags

def findMostLikelyTags(trainCorpus):
    totalTags = []
    Ttags = brown.tagged_sents()

    for i in range((len(trainCorpus))):
        tags = findTagsOf(trainCorpus[i], Ttags[i])
        totalTags += tags

    #counting the count of tags in porpus using dictionary

    dict = {}
    # the dict contains for each word a tuple of {tag : amount}==> tag is the word' tag and its amount of time in corpus
    for i in range(len(totalTags)):
        word = totalTags[i][0]
        tag  =totalTags[i][1]

        if word in dict:
            if tag not in dict[word]:
                dict[word][tag] = 1
            else:
                dict[word][tag] += 1
        else:
            dict[word] = {tag : 1}

    #for each word get a tag with the max amount in the dict
    finalArray = {}

    for word, tag in dict.items():
        v = list(tag.values())
        k = list(tag.keys())
        finalTag  = k[v.index(max(v))]
        finalArray[word] = finalTag
    return finalArray



def ErrorRateForTest(testCorpus, mostLikelyTags):

    truePredicts = 0
    totalPredicts = 1
    unknownTruePredicts = 0
    totalUnkownPredicts = 1
    known_words = {}
    unKnown_words = {}

    for i in range(len(testCorpus)):
        viterbi = False
        IthSentence = testCorpus[i]

        if type(IthSentence) is tuple:
            viterbi = True
            word = IthSentence[0]
            tag = IthSentence[1]
            """            
            if tag == mostLikelyTags[i] and mostLikelyTags[i] != 'NN':
                totalPredicts += 1
                truePredicts += 1
            elif  mostLikelyTags[i] == 'NN':
                if tag == mostLikelyTags[i]:
                    unknownTruePredicts += 1
                totalUnkownPredicts += 1
            else:
                totalPredicts += 1
                """
            if tag == 'NN':
                if mostLikelyTags[i] == 'NN':
                    unknownTruePredicts += 1
                totalUnkownPredicts += 1
            else:
                if mostLikelyTags[i] == tag:
                    truePredicts += 1
                totalPredicts += 1

        else:

            for k in range(len(IthSentence)):
                word = IthSentence[k][0]
                tag = IthSentence[k][1]

                if word in mostLikelyTags:
                    if word in known_words:
                        known_words[word]["num_of_shows"] += 1
                        if tag == mostLikelyTags[word]:
                            known_words[word]["correctTags"] += 1
                            truePredicts += 1
                    if word not in known_words:
                        if tag == mostLikelyTags[word]:
                            known_words[word] = {"num_of_shows":1 , "correctTags" : 1}
                            truePredicts += 1
                        else:
                            known_words[word] = {"num_of_shows":1 , "correctTags" : 0}
                    totalPredicts += 1

                else:# if the is unKnownWords
                    if word in unKnown_words:
                        unKnown_words[word]["num_of_shows"] += 1
                        if tag == 'NN': #unknown tag
                            unknownTruePredicts += 1
                            unKnown_words[word]["correctTags"] += 1
                    else:
                        if tag == 'NN':
                            unKnown_words[word] = {"num_of_shows" :1, "correctTags" : 1}
                            unknownTruePredicts += 1
                        else:
                            unKnown_words[word] = {"num_of_shows": 1, "correctTags": 0}
                    totalUnkownPredicts += 1

    erroRateKnownWords = 1 - truePredicts/totalPredicts
    erroRateUknownWords = 1 - unknownTruePredicts/totalUnkownPredicts
    totalP = totalPredicts + totalUnkownPredicts
    totalTrueP = truePredicts + unknownTruePredicts
    totalErroRate = 1 - totalTrueP/totalP

    return erroRateKnownWords , erroRateUknownWords, totalErroRate


def transitionProbs(trainCorpus):

    qDict = defaultdict(Counter)

    for i in range(len(trainCorpus)):
        IthSentence = trainCorpus[i]
        priorTag = "*"

        for word, tag in IthSentence:
            if priorTag not in qDict.keys():
                qDict[priorTag] = {tag : 1}
            elif tag in qDict[priorTag]:
                qDict[priorTag][tag] += 1
            else:
                qDict[priorTag][tag] = 1
            priorTag = tag

        if 'STOP' not in qDict[priorTag]:
            qDict[priorTag] = {'STOP' : 1}

        else:
            qDict[priorTag]['STOP'] += 1
    return qDict



def emissionProbs(trainCorpus):

    #use defaultkey collection to support duplicate keys
    eDict = defaultdict(Counter)

    for i in range(len(trainCorpus)):
        IthSentence = trainCorpus[i]
        priorTag = "*"

        for word, tag in IthSentence:

            if tag not in eDict.keys():
                eDict[tag] = {word :1}
            elif word in eDict[tag]:#####
                eDict[tag][word] += 1
            else:
                eDict[tag][word] = 1
            priorTag = tag

        if 'STOP' not in eDict[priorTag]:
            eDict[priorTag] = {'STOP' : 1}
        else:
            eDict[priorTag]['STOP'] += 1

    return eDict

def viterbiAlg(sentence,emissionPDict, transitionPDict,StagsList,algorithm):
    """
    :param sentence : the sentence from the train corpus on which the Alg' returns most likely tags
    :param emissionPDict : dictionary with emission probabilities
    :param transitionPDict : dictionary with transition probabilities
    :param StagsList : set of tags from training corpus
    :return : sequence of tags corresponding to most likely tags of the given sentence
    """
    #wordsList = word_tokenize(sentence)
    wordsList = sentence
    piDict = defaultdict(Counter)
    #stores for each sequence location,n,and tag, itag1, its pi value,
    #given the previous n-1 pi value

    #initialize "pi"-->
    piDict[0]['*'] = 1

    tagsDict = {}
    #stores for each sequence location n ,and tag itag1, the former tag within the sequence,
    #that comes before it which succeeds the maximal  value, of the corresponding piDict[n] value

    for n in range(1,len(wordsList)+1):

        tagsDict[n] = {}
        for itag1 in StagsList:
            #compute emisionProbs->
            emissionP = SeqentialProb(wordsList[n-1],itag1,emissionPDict,algorithm,"emission")


            if n-1 == 0:#we are on first tag within the sequence
                piDict[n][itag1] = SeqentialProb(itag1,'*',transitionPDict,algorithm,"trans")*emissionP*piDict[0]['*']
                tagsDict[n][itag1] = '*'

            else:

                #for a bigram HMM model iterate over the previous tag only
                maxPi = 0
                maxTag = None
                for itag2 in StagsList:
                    #compute transition probailiy from each tag to another
                    transitionP = SeqentialProb(itag1,itag2,transitionPDict,algorithm,"trans")

                    #check for debug!!!#todo omit before submitting!
                    if(emissionP>0 and transitionP >0):
                        isZero = piDict[n-1][itag2]
                        #print(isZero)


                    #search for the maximum value!!
                    currentMax = piDict[n-1][itag2]*emissionP*transitionP
                    if currentMax > maxPi:
                        maxPi = currentMax
                        maxTag = itag2
                    #unknown tag -->
                if maxTag == None:
                    maxTag = 'NN'
                piDict[n][itag1] = maxPi
                tagsDict[n][itag1] = maxTag

    #compute all sequence tags from the above dicts
    #start by the last tag, for the most likely tag before it
    # is given by tagsDict[n][lastTag]-->
    maxYnProb = -1
    lastTag = None
    for tag in StagsList:
        currentMax = piDict[len(sentence)][tag]*SeqentialProb('STOP',tag,transitionPDict,algorithm,"trans")
        if currentMax > maxYnProb:
            maxYnProb = currentMax
            lastTag = tag
    tagsArray = []
    currenTag = lastTag
    tagsArray.append(currenTag)
    for i in range(len(sentence) - 1,0,-1):
        newTag = tagsDict[i+1][currenTag]
        tagsArray.append(newTag)
        currenTag = newTag


    tagsArray.reverse()
    return tagsArray

def compEmissionProbAddSmooth(labelTag,word,emissionDict):

    return (emissionDict[labelTag][word] +1)/(sum(emissionDict[labelTag].values() + trainSizeCorpus))


def SeqentialProb(token1, token2,dictValues,algorithm,case):
    """compute p(token1 | token2) using dictionary"""
    partial = 0
    sum1 = 0

    if case == "trans" or (case == "emission" and algorithm == BIGRAM_HMM):

        sum1 = sum(dictValues[token2].values())
        if token1 not in dictValues[token2]:
            partial = 0
        else:
            partial = dictValues[token2][token1]

    elif case == "emission":

        if algorithm == ADD_SMOOTH:
            sum1 = sum(dictValues[token2].values()) + trainSizeCorpus ########## maybe omit "+ trainSizeCorpus" #todo
            if token1 not in dictValues[token2]:
                partial = 0#unknowm words still have 0 prob.
            else:
                partial = dictValues[token2][token1] + 1

        if algorithm == PSEUDOWORDS:
            sum1 = sum(dictValues[token2].values())
            if token1 not in dictValues[token2]:
                token1 = replaceWithPseudoWords(token1)
            if token1 not in dictValues[token2]:
                partial = 0
            else:
                partial = dictValues[token2][token1]

        if algorithm == PSEUDOWORDSANDSMOOTH:
            sum1 = sum(dictValues[token2].values())+trainSizeCorpus  ########## maybe omit "+ trainSizeCorpus" #todo
            if token1 not in dictValues[token2]:
                token1 = replaceWithPseudoWords(token1)
            if token1 not in dictValues[token2]:
                partial = 0
            else:
                partial = dictValues[token2][token1] + 1

    return partial/sum1


def emissionDictForPSEUDOW(eDICT):
    """replace all words which appear less then MINFREQUENCY times in the emission Dict with PSEUDOWORDS"""
    newEmissionDict = defaultdict(Counter)
    for tag in eDICT:
        for word in eDICT[tag]:
            newWord = replaceWithPseudoWords(word)

            if eDICT[tag][word] >= MINFREQUENCY:
                if newWord in newEmissionDict[tag]:
                    newEmissionDict[tag][newWord] += eDICT[tag][word]
                else:
                    newEmissionDict[tag][newWord] = eDICT[tag][word]

            else:

                if newWord in newEmissionDict[tag]:
                    newEmissionDict[tag][newWord] += eDICT[tag][word]
                else:
                    newEmissionDict[tag][newWord] = eDICT[tag][word]
    return newEmissionDict


def replaceWithPseudoWords(word):
    for pattern in PSEUDOWORDSDICT.keys():
        if re.findall(pattern,word):
            return PSEUDOWORDSDICT[pattern]

    return word





def runTests(transitionDict,emissionDict,StagsArray,algorithm):

    if algorithm == BIGRAM_HMM:
        print("")
        print("------------>HMM bigram model for POS<-------------")
    if algorithm == ADD_SMOOTH:
        print("")
        print("------------>HMM Add-one smoothing for POS<-------------")
    if algorithm == PSEUDOWORDS:
        print("")
        print("------------>HMM Pseudo-Words for POS<-------------")
    if algorithm == PSEUDOWORDSANDSMOOTH:
        print("")
        print("------------>HMM Pseudo-Words and SMOOTHING for POS<-------------")

    sumAllTestsErRatesKnown = 0
    sumAllTestsErRatesUnkown = 0
    sumAllTestsErRatesTotal = 0
    count = 0
    labelList = []
    predictedList  = []

    for testSentence in testCorpus:

        count += 1
        sentence = [word for word,tag in testSentence]

        tagList = viterbiAlg(sentence,emissionDict,transitionDict,StagsArray,algorithm)
        predictedList.extend(tagList)
        labelList.extend([tag for word,tag in testSentence])
        erKnown, erUnKnown,totErate = ErrorRateForTest(testSentence,tagList)
        sumAllTestsErRatesKnown += erKnown
        sumAllTestsErRatesUnkown += erUnKnown
        sumAllTestsErRatesTotal += totErate

    sumAllTestsErRatesKnown = sumAllTestsErRatesKnown/count
    sumAllTestsErRatesUnkown = sumAllTestsErRatesUnkown/count
    sumAllTestsErRatesTotal = sumAllTestsErRatesTotal/count
    if(algorithm == PSEUDOWORDSANDSMOOTH):
        print("confusion Matrix for - ", algorithm)
        print("")
        confusionMatrix = nltk.ConfusionMatrix(labelList,predictedList)
        print(confusionMatrix.pretty_format(sort_by_count = True,show_percents = False))
    print("-errorRateTest - Known : ",sumAllTestsErRatesKnown )
    print("-errorRateTest - UnKnown : ",sumAllTestsErRatesUnkown )
    print("-errorRateTest - total : ",sumAllTestsErRatesTotal)
    print("")
    print("--------------------------------------------")



mostLikelyWord = findMostLikelyTags(trainCorpus)
eRateKnown , eRateUnknown, totalErate = ErrorRateForTest(testCorpus,mostLikelyWord)
print("")
print("------------> logLiklihood model for POS<----------")
print("-errorRateTest - Known : ",eRateKnown )
print("-errorRateTest - UnKnown : ",eRateUnknown )
print("-errorRateTest - total : ",totalErate )
eDICT = emissionProbs(trainCorpus)
qDiict = transitionProbs(trainCorpus)
Stags  = [tag for tag in transitionProbs(trainCorpus)]
Stags.remove('*')

for algorithm in [BIGRAM_HMM, ADD_SMOOTH, PSEUDOWORDS, PSEUDOWORDSANDSMOOTH]:

    if algorithm == PSEUDOWORDS or algorithm == PSEUDOWORDSANDSMOOTH:
        eDICT = emissionDictForPSEUDOW(eDICT)
runTests(qDiict,eDICT,Stags,algorithm)
