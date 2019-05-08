from nltk.collocations import *
from collections import defaultdict
from nltk.corpus import names, wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import text
from nltk import word_tokenize
from collections import Counter
from nltk.util import ngrams

import glob
import os
import numpy as np
import nltk
import string
import re
import time
import pandas as pd
import xlrd

################################## VOCAB-FOCUSED TERM FREQUENCY APIs ##################################


def hasApprovedPos(partOfSpeechTag):
    if(partOfSpeechTag == 'CD' or partOfSpeechTag == 'JJ' or partOfSpeechTag == 'JJR' or partOfSpeechTag == 'JJS' or partOfSpeechTag == 'MD' or
    partOfSpeechTag == 'RB' or partOfSpeechTag == 'RBR' or partOfSpeechTag == 'RBS'):
        return True
    return False

def generateVocabSet(filename):
    text = extractSingleTextFile(filename)
    vocabset = []
    for word in text.split():
        pos = nltk.pos_tag(nltk.word_tokenize(word.lower()))
        if hasApprovedPos(pos[0][len(pos[0])-1]):
            vocabset.append(pos[0][0])

    return set(vocabset)



################################## ORIGINAL TERM FREQUENCY APIs ##################################

def extendedPOSList(partOfSpeechTag):
    if (partOfSpeechTag == 'CC' or partOfSpeechTag == 'PRP$' or partOfSpeechTag == 'PRP' or
            partOfSpeechTag == 'UH' or partOfSpeechTag == 'WP' or partOfSpeechTag == 'WP$' or
            partOfSpeechTag == 'IN'):
        return True
    return False

def importAllTextFilesAndGenerateLabels(folderScam, folderNotScam, directory):
    textfiles = []
    labels = []
    os.chdir(directory)
    for filename in glob.glob(os.path.join(folderScam, '*.txt')):
        textfiles.append(extractSingleTextFile(filename))
        label.append(1)

    for filename in glob.glob(os.path.join(folderNotScam, '*.txt')):
        textfiles.append(extractSingleTextFile(filename))
        labels.append(0)

    return textfiles, labels

def generateNewStopWordsList():
    newStopList = {" "}
    for stopWord in text.ENGLISH_STOP_WORDS:
        wordChoice = nltk.word_tokenize(stopWord)
        pos = nltk.pos_tag(wordChoice)
        if extendedPOSList(pos[0][len(pos[0]) - 1]):
            newStopList.add(stopWord)
    return frozenset(newStopList)

def cleanText(docs):
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    cleanedFiles = []

    for doc in docs:
        cleanedFiles.append(
            ' '.join([lemmatizer.lemmatize(word.lower())
                      for word in doc.split()
                        if word.isalpha() and word not in all_names])
        )

    return cleanedFiles

def generateUltimateSet(vocabset):
    ultimateSet = []
    for pos in vocabset:
        syn = wn.synsets(pos)
        for syno in syn[0]._lemma_names:
            ultimateSet.append(syno)
        ultimateSet.append(pos)

    return set(ultimateSet)

def returnFilesSplit(splitList):
    return splitList[0], splitList[1]

def returnLabelsSplit(splitList):
    return splitList[2], splitList[3]

def getNltkNewStopWordsList():
    newStopList = {" "}
    for stopWord in stopwords.words('english'):
        wordChoice = nltk.word_tokenize(stopWord)
        pos = nltk.pos_tag(wordChoice)
        if extendedPOSList(pos[0][len(pos[0]) - 1]):
            newStopList.add(stopWord)
    return frozenset(newStopList)

def getLabelIndex(labels):
    ##gets the indices of all files labelled 1 and 0 and puts them into a dictionary
    ##where 1 and 0 are the keys and the values for each are lists of the indices
    labelIndex = defaultdict(list)
    for index, label in enumerate(labels):
        labelIndex[label].append(index)
    return labelIndex

def getLowercaseNames():
    lowercaseNames = {" "}
    for name in names.words():
        lowercaseNames.add(name.lower())
    return lowercaseNames

def getPrior(labelIndex):
    #each label has length of index
    prior = {label: len(index) for label, index in labelIndex.items()}

    #.items() lists all the pairs in the dictionary
    #.values() lists the values for every pair in the dictionary

    totalCount = sum(prior.values()) #the total number of values
    for label in prior:
        prior[label] /= float(totalCount)
    #divides everything by the total number of values
    return prior

def generateCountVectorizer(stopWords, maxFeatures):
    return CountVectorizer(stop_words=stopWords, max_features=maxFeatures)

def generateTrainTermDocMatrix(trainSplit, cv):
    return cv.fit_transform(trainSplit)

def generateTestTermDocMatrix(testSplit, cv):
    return cv.transform(testSplit)

def getLikelihood(termDocumentMatrix, labelIndex, smoothing=1):

    likelihood = {}
    # print(labelIndex.items()) a list of all the items (dictionary) (array of emails numbers for each label)
    for label, index in labelIndex.items():

        ##below sums values at each row(email index) and adds 1 for smoothing - P(S,features) - all divisible by the same number
        ##it's a sparse matrix so some of the values are 0
        likelihood[label] = termDocumentMatrix[index, :].sum(axis=0) + smoothing #sums all the values

        likelihood[label] = np.asarray(likelihood[label])[0]

        totalCount = likelihood[label].sum()
        likelihood[label] = likelihood[label]/float(totalCount)

    return likelihood

def getPosterior(termDocs, prior, likelihood):
    numDocs = termDocs.shape[0]
    posteriors = []

    for i in range(numDocs):
        posterior = {key: np.log(priorLabel)
                     for key, priorLabel in prior.items()}
        for label, likelihoodLabel in likelihood.items():
            termDocsVector = termDocs.getrow(i)
            counts = termDocsVector.data
            indices = termDocsVector.indices

            for count,index in zip(counts, indices):
                posterior[label] += np.log(likelihoodLabel[index]) * count

        minLogPosterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - minLogPosterior)
            except:
                posterior[label] = float('inf')

        sumPosterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sumPosterior
        posteriors.append(posterior.copy())
    return posteriors

def get_posterior(tdm, prior, likelihood):
    num_docs = tdm.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label)
                     for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            tdv = tdm.getrow(i)
            counts = tdv.data
            indices = tdv.indices

            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index])*count

        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

def generateTermFrequencyPredictions():

    textfiles, labels = importAllTextFilesAndGenerateLabels('enron1/rawTextScam/', 'enron1/rawTextNotScam/', "C:/Users/mpalu/PycharmProjects/practiceML")
    cleanedText = cleanedText(textfiles)
    labelIndex = getLabelIndex(labels)

    splitList = train_test_split(cleanedText, labels, test_size=0.25, random_state=42)

    textTrain, textTest = returnFilesSplit(splitList)
    labelsTrain, labelsTest = returnLabelsSplit(splitList)

    cv = generateCountVectorizer('english', 25)
    termDocsTrain = generateTrainTermDocMatrix(textTrain, cv)

    prior = getPrior(labelIndex)
    likelihood = getLikelihood(termDocsTrain, labelIndex, 1)

    termDocsTest = generateTestTermDocMatrix(textTest, cv)

    return getPosterior(termDocsTest, prior, likelihood)

def printPredictions(posterior, yTest):
    outputFile = open("enron1/testFile.txt", 'w')
    correct = 0.0
    correctPrediction = "correct prediction"
    incorrectPrediction = "incorrect prediction"

    for pred, actual in zip(posterior, yTest):
        if actual == 1:
            if pred[1] >= 0.5:
                correct += 1
                outputFile.write(correctPrediction + ", actual result: " + str(actual) + ", rounded prediction " + str(1) + ", actual prediction " + str(pred[1]) + "\n")

                print(correctPrediction + " actual result: " + str(actual) + " rounded prediction " + str(1) + ", actual prediction " + str(pred[1]))
            else:
                outputFile.write(incorrectPrediction + ", actual result: " + str(actual) + ", prediction " + str(0) + ", actual prediction " + str(pred[0]) + "\n")

                print(incorrectPrediction + " actual result: " + str(actual) + " prediction " + str(0) + ", actual prediction " + str(pred[0]))
        else:
            if pred[0] > 0.5:
                correct += 1
                outputFile.write(correctPrediction + ", actual result: " + str(actual) + ", prediction " + str(0) + ", actual prediction " + str(pred[0]) + "\n")

                print(correctPrediction + " actual result: " + str(actual) + " prediction " + str(0) + ", actual prediction " + str(pred[0]))
            else:
                outputFile.write(incorrectPrediction + ", actual result:  " + str(actual) + ", prediction " + str(1) + ", actual prediction " + str(pred[1]) + "\n")

                print(incorrectPrediction + " actual result:  " + str(actual) + " prediction " + str(1) + ", actual prediction " + str(pred[1]))
    print('The accuracy on {0} testing samples is:'
          '{1:.1f}%'.format(len(yTest), correct / len(yTest) * 100))

class Vocab:
    def __init__(self, scamFilesDirectory, notScamFilesDirectory, vocabFile):
        self.vocabList = {}

        self.directory = r"C:\Users\mpalu\PycharmProjects\practiceML"
        self.scamFilesDirectory = scamFilesDirectory
        self.notScamFilesDirectory = notScamFilesDirectory
        self.vocabFile = vocabFile      ## is an excel file

        self.unigramList = {}
        self.bigramList = {}
        self.trigramList = {}

        self.cleanedFiles = []
        self.vocabOnlyFile = []

        self.labels = []
        self.vocabLabels = []
        self.labelIndex = []
        self.vocabLabelIndex = []

        self.trainFiles = []
        self.trainLabels = []
        self.testFiles = []
        self.testLabels = []

        self.vocabTrainFiles = []
        self.vocabTrainLabels = []
        self.vocabTestFiles = []
        self.vocabTestLabels = []

        self.vocab_cv = CountVectorizer()
        self.cv = CountVectorizer(stop_words='english', max_features=31)

        self.vocab_termDocsTrain = []
        self.termDocsTrain = []

        self.vocab_termDocsTest = []
        self.termDocsTest = []

        self.vocabPrior = {}
        self.prior = {}

        self.vocabLikelihood = {}
        self.likelihood = {}

        self.vocabPosteriors = []
        self.posteriors = []

    def extractSingleTextFile(self, filename):
        with open(filename, 'r', encoding="utf-8") as infile:
            s = infile.read()
            return s.translate(str.maketrans('', '', string.punctuation))

    def importAllAndCleanTextFilesAndGenerateLabels(self):

        all_names = set(names.words())
        lemmatizer = WordNetLemmatizer()

        os.chdir(self.directory)
        for filename in glob.glob(os.path.join(self.scamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            self.cleanedFiles.append(' '.join([lemmatizer.lemmatize(word.lower())
                                            for word in doc.split()
                                            if word.isalpha() and word not in all_names]))
            self.labels.append(1)
        for filename in glob.glob(os.path.join(self.notScamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            self.cleanedFiles.append(' '.join([lemmatizer.lemmatize(word.lower())
                                            for word in doc.split()
                                            if word.isalpha() and word not in all_names]))
            self.labels.append(0)

    def getTrainingSplit(self):
        self.trainFiles, self.testFiles, self.trainLabels, self.testLabels = train_test_split(self.cleanedFiles, self.labels, test_size=0.25, random_state=42)

    def getLabelIndex(self):
        ##gets the indices of all files labelled 1 and 0 and puts them into a dictionary
        ##where 1 and 0 are the keys and the values for each are lists of the indices
        self.labelIndex = defaultdict(list)
        for index, label in enumerate(self.trainLabels):
            self.labelIndex[label].append(index)

    def getPrior(self):
        # each label has length of index
        self.prior = {label: len(index) for label, index in self.labelIndex.items()}

        # .items() lists all the pairs in the dictionary
        # .values() lists the values for every pair in the dictionary

        totalCount = sum(self.prior.values())  # the total number of values
        for label in self.prior:
            self.prior[label] /= float(totalCount)
        # divides everything by the total number of values

    def getLikelihood(self, smoothing=1):
        for label, index in self.labelIndex.items():
            ##below sums values at each row(email index) and adds 1 for smoothing - P(S,features) - all divisible by the same number
            ##it's a sparse matrix so some of the values are 0
            self.likelihood[label] = self.termDocsTrain[index, :].sum(axis=0) + smoothing  # sums all the values

            self.likelihood[label] = np.asarray(self.likelihood[label])[0]

            totalCount = self.likelihood[label].sum()
            self.likelihood[label] = self.likelihood[label] / float(totalCount)

    def generateTrainedModel(self):
        self.termDocsTrain = self.cv.fit_transform(self.trainFiles)
        self.getPrior()
        self.getLikelihood()

    def generatePosterior(self):
        self.termDocsTest = self.cv.transform(self.testFiles)
        self.get_posterior()

    def get_posterior(self):
        num_docs = self.termDocsTest.shape[0]
        for i in range(num_docs):
            posterior = {key: np.log(prior_label)
                         for key, prior_label in self.prior.items()}
            for label, likelihood_label in self.likelihood.items():
                tdv = self.termDocsTest.getrow(i)
                counts = tdv.data
                indices = tdv.indices

                for count, index in zip(counts, indices):
                    posterior[label] += np.log(likelihood_label[index]) * count

            min_log_posterior = min(posterior.values())
            for label in posterior:
                try:
                    posterior[label] = np.exp(posterior[label] - min_log_posterior)
                except:
                    posterior[label] = float('inf')
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
            self.posteriors.append(posterior.copy())

    def printPosterior(self):
        correct = 0.0
        correctPrediction = "correct prediction"
        incorrectPrediction = "incorrect prediction"

        for pred, actual in zip(self.posteriors, self.testLabels):
            if actual == 1:
                if pred[1] >= 0.5:
                    correct += 1
                    print(correctPrediction + "actual: " + str(actual) + " rounded prediction " + str(
                        1) + ", actual prediction " + str(pred[1]))
                else:
                    print(incorrectPrediction + "actual: " + str(actual) + " prediction " + str(
                        0) + ", actual prediction " + str(pred[0]))
            else:
                if pred[0] > 0.5:
                    correct += 1
                    print(correctPrediction + "actual: " + str(actual) + " prediction " + str(
                        0) + ", actual prediction " + str(pred[0]))
                else:
                    print(incorrectPrediction + "actual: " + str(actual) + " prediction " + str(
                        1) + ", actual prediction " + str(pred[1]))
        print('The accuracy on {0} testing samples is:'
              '{1:.1f}%'.format(len(self.testLabels), correct / len(self.testLabels) * 100))

    def returnResult(self):
        self.importAllAndCleanTextFilesAndGenerateLabels()
        self.getTrainingSplit()
        self.getLabelIndex()
        self.generateTrainedModel()

        multi = MultinomialNB(alpha=1, fit_prior=True)
        multi.fit(self.termDocsTrain, self.trainLabels)
        self.termDocsTest = self.cv.transform(self.testFiles)
        prediction = multi.predict_proba(self.termDocsTest)
        print("Accuracy: {}".format(multi.score(self.termDocsTest, self.testLabels)))
        print("Summary:\nNumber of training files: {0}\nNumber of testing files {1}".format(len(self.trainFiles),
                                                                                            len(self.testFiles)))

##################################### VOCAB ##################################

    def importAllAndExtractVocabAndGenerateLabels(self):

        lemmatizer = WordNetLemmatizer()

        os.chdir(self.directory)
        for filename in glob.glob(os.path.join(self.scamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            uni = (' '.join([str(key[0].lower())
                    for key in Counter(ngrams(nltk.word_tokenize(doc), 1))
                    if str(key[0].lower()) in self.unigramList]))

            bi = (' '.join([str(key[0].lower() + " " + key[1].lower())
                    for key in Counter(ngrams(nltk.word_tokenize(doc), 2))
                    if str(key[0].lower() + " " + key[1].lower()) in self.bigramList]))
            tri = (' '.join([str(key[0].lower() + " " + key[1].lower() + " " + key[2].lower())
                    for key in Counter(ngrams(nltk.word_tokenize(doc), 3))
                    if str(key[0].lower() + " " + key[1].lower() + " " + key[2].lower()) in self.trigramList]))
            self.vocabOnlyFile.append(uni + " " + bi + " " + tri)
            self.vocabLabels.append(1)
        for filename in glob.glob(os.path.join(self.notScamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            uni = (' '.join([str(key[0].lower())
                             for key in Counter(ngrams(nltk.word_tokenize(doc), 1))
                             if str(key[0].lower()) in self.unigramList]))

            bi = (' '.join([str(key[0].lower() + " " + key[1].lower())
                            for key in Counter(ngrams(nltk.word_tokenize(doc), 2))
                            if str(key[0].lower() + " " + key[1].lower()) in self.bigramList]))
            tri = (' '.join([str(key[0].lower() + " " + key[1].lower() + " " + key[2].lower())
                             for key in Counter(ngrams(nltk.word_tokenize(doc), 3))
                             if str(key[0].lower() + " " + key[1].lower() + " " + key[2].lower()) in self.trigramList]))
            self.vocabOnlyFile.append(uni + " " + bi + " " + tri)
            self.vocabLabels.append(0)

    def importVocabList(self):
        self.unigramList= set(pd.read_excel(self.vocabFile, sheet_name='unigrams')["NAMES"])
        self.bigramList = set(pd.read_excel(self.vocabFile, sheet_name='bigrams')["NAMES"])
        self.trigramList = set(pd.read_excel(self.vocabFile, sheet_name='trigrams')["NAMES"])

    def getVocabLabelIndex(self):
        self.vocabLabelIndex = defaultdict(list)
        for index, label in enumerate(self.vocabTrainLabels):
            self.vocabLabelIndex[label].append(index)

    def getVocabTrainingSplit(self):
        self.vocabTrainFiles, self.vocabTestFiles, self.vocabTrainLabels, self.vocabTestLabels = train_test_split(self.vocabOnlyFile, self.vocabLabels, test_size=0.25, random_state=42)

    def getVocabPrior(self):
        self.vocabPrior = {label: len(index) for label, index in self.vocabLabelIndex.items()}

        totalCount = sum(self.vocabPrior.values())
        for label in self.vocabPrior:
            self.vocabPrior[label] /= float(totalCount)

    def getVocabLikelihood(self, smoothing=1):
        for label, index in self.vocabLabelIndex.items():
            ##below sums values at each row(email index) and adds 1 for smoothing - P(S,features) - all divisible by the same number
            ##it's a sparse matrix so some of the values are 0
            self.vocabLikelihood[label] = self.vocab_termDocsTrain[index, :].sum(axis=0) + smoothing  # sums all the values

            self.vocabLikelihood[label] = np.asarray(self.vocabLikelihood[label])[0]

            totalCount = self.vocabLikelihood[label].sum()
            self.vocabLikelihood[label] = self.vocabLikelihood[label] / float(totalCount)

    def generateTrainedVocabModel(self):
        self.vocab_termDocsTrain = self.vocab_cv.fit_transform(self.vocabTrainFiles)
        self.getVocabPrior()
        self.getVocabLikelihood()

    def getVocabPosterior(self):
        num_docs = self.vocab_termDocsTest.shape[0]
        for i in range(num_docs):
            posterior = {key: np.log(prior_label)
                         for key, prior_label in self.vocabPrior.items()}
            for label, likelihood_label in self.vocabLikelihood.items():
                tdv = self.vocab_termDocsTest.getrow(i)
                counts = tdv.data
                indices = tdv.indices

                for count, index in zip(counts, indices):
                    posterior[label] += np.log(likelihood_label[index]) * count

            min_log_posterior = min(posterior.values())
            for label in posterior:
                try:
                    posterior[label] = np.exp(posterior[label] - min_log_posterior)
                except:
                    posterior[label] = float('inf')
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
            self.vocabPosteriors.append(posterior.copy())

    def generateVocabPosterior(self):
        self.vocab_termDocsTest = self.vocab_cv.fit_transform(self.vocabTestFiles)
        self.getVocabPosterior()

    def printVocabPosterior(self):
        correct = 0.0
        correctPrediction = "correct prediction"
        incorrectPrediction = "incorrect prediction"

        for pred, actual in zip(self.vocabPosteriors, self.vocabTestLabels):
            if actual == 1:
                if pred[1] >= 0.5:
                    correct += 1
                    print(correctPrediction + "actual: " + str(actual) + " rounded prediction " + str(
                        1) + ", actual prediction " + str(pred[1]))
                else:
                    print(incorrectPrediction + "actual: " + str(actual) + " prediction " + str(
                        0) + ", actual prediction " + str(pred[0]))
            else:
                if pred[0] > 0.5:
                    correct += 1
                    print(correctPrediction + "actual: " + str(actual) + " prediction " + str(
                        0) + ", actual prediction " + str(pred[0]))
                else:
                    print(incorrectPrediction + "actual: " + str(actual) + " prediction " + str(
                        1) + ", actual prediction " + str(pred[1]))
        print('VOCAB: The accuracy on {0} testing samples is:'
              '{1:.1f}%'.format(len(self.vocabTestLabels), correct / len(self.vocabTestLabels) * 100))

    def returnVocabResult(self):

        self.importVocabList()
        self.importAllAndExtractVocabAndGenerateLabels()
        self.getVocabTrainingSplit()
        self.getVocabLabelIndex()
        self.generateTrainedVocabModel()

        # self.generateVocabPosterior()
        # self.printVocabPosterior()

        multi = MultinomialNB(alpha=1, fit_prior=True)
        multi.fit(self.vocab_termDocsTrain, self.vocabTrainLabels)
        self.vocab_termDocsTest = self.vocab_cv.transform(self.vocabTestFiles)
        print("Vocab Accuracy: {}".format(multi.score(self.vocab_termDocsTest, self.vocabTestLabels)))

        print("Summary:\nNumber of training files: {}\nNumber of testing files {}".format(len(self.vocabTrainFiles),
                                                                                          len(self.vocabTestFiles)))

def main():

    t0 = time.time()
    os.chdir(r"C:\Users\mpalu\PycharmProjects\practiceML")
    vocabObj = Vocab(r"enron1\rawTextScam",r"enron1\rawTextNotScam",r"vocab.xlsx")

    print("VOCAB")
    vocabObj.returnVocabResult()
    print("NOT VOCAB")
    vocabObj.returnResult()

    t1 = time.time()
    print("TOTAL TIME: " + str((t1-t0)) + "SECONDS")

if __name__=="__main__":
    main()
