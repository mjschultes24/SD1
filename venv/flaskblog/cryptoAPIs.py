from nltk.collocations import *
from collections import defaultdict
from nltk.corpus import names, wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

class Vocab:
    def __init__(self, scamFilesDirectory, notScamFilesDirectory, testFilesDirectory, vocabFile):
        self.vocabList = {}

        self.directory = r"C:\Users\mpalu\PycharmProjects\practiceML"
        self.scamFilesDirectory = scamFilesDirectory
        self.notScamFilesDirectory = notScamFilesDirectory
        self.testFilesDirectory = testFilesDirectory
        self.vocabFile = vocabFile      ## is an excel file

        self.unigramList = {}
        self.bigramList = {}
        self.trigramList = {}

        self.cleanedFiles = []
        self.cleanedTestFiles = []
        self.vocabOnlyFile = []
        self.vocabOnlyTestFiles = []

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
        
    def returnFilesResult(self):
        self.importAllAndCleanTextFilesAndGenerateLabels()
        self.getTrainingSplit()
        self.getLabelIndex()
        self.generateTrainedModel()
        multi = MultinomialNB(alpha=1, fit_prior=True)
        multi.fit(self.termDocsTrain, self.trainLabels)
        self.termDocsTest = self.cv.transform(self.testFiles)
        prediction = multi.predict_proba(self.termDocsTest)
        return prediction

##################################### VOCAB ##################################

    def importAllAndExtractVocabAndGenerateLabels(self):

        lemmatizer = WordNetLemmatizer()

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

    def importAllTestFilesAndExtractVocab(self):

        lemmatizer = WordNetLemmatizer()

        for filename in glob.glob(os.path.join(self.testFilesDirectory, '*.txt')):
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
            self.vocabOnlyTestFiles.append(uni + " " + bi + " " + tri)
    def importVocabList(self):
        self.unigramList = set(pd.read_excel(self.vocabFile, sheet_name='unigrams')["NAMES"])
        self.bigramList = set(pd.read_excel(self.vocabFile, sheet_name='bigrams')["NAMES"])
        self.trigramList = set(pd.read_excel(self.vocabFile, sheet_name='trigrams')["NAMES"])

    def getVocabLabelIndex(self):
        self.vocabLabelIndex = defaultdict(list)
        for index, label in enumerate(self.vocabOnlyFile):
            self.vocabLabelIndex[label].append(index)

    # def getVocabTrainingSplit(self):
    #     self.vocabTrainFiles, self.vocabTestFiles, self.vocabTrainLabels, self.vocabTestLabels = train_test_split(self.vocabOnlyFile, self.vocabLabels, test_size=0.25, random_state=42)

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
        self.vocab_termDocsTrain = self.vocab_cv.fit_transform(self.vocabOnlyFile)

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

    def vocabPreprocessingStep(self):
        self.importVocabList()
        self.importAllAndExtractVocabAndGenerateLabels()

    def returnVocabFilesResult(self):
        self.vocabPreprocessingStep()
        self.getVocabLabelIndex()
        self.generateTrainedVocabModel()
        multi = MultinomialNB(alpha=1, fit_prior=True)
        multi.fit(self.vocab_termDocsTrain, self.vocabLabels)
        self.importAllTestFilesAndExtractVocab()
        self.vocab_termDocsTest = self.vocab_cv.transform(self.vocabOnlyTestFiles)
        predictions = multi.predict_proba(self.vocab_termDocsTest)
        ##gives answer in not scam, scam
        return predictions

def main():

    t0 = time.time()

    #scamFilesDirectory, notScamFilesDirectory, testFilesDirectory, vocabFile
    scamFilesDirectory = r"C:\Users\mpalu\PycharmProjects\practiceML\enron1\rawTextScam"
    notScamFilesDirectory = r"C:\Users\mpalu\PycharmProjects\practiceML\enron1\rawTextNotScam"
    testFilesDirectory = r"C:\Users\mpalu\PycharmProjects\practiceML\enron1\test"
    vocabDirectory = r"C:\Users\mpalu\PycharmProjects\practiceML\vocab.xlsx"

    vocabObj = Vocab(scamFilesDirectory,notScamFilesDirectory,testFilesDirectory,vocabDirectory)



    print("VOCAB")
    print(vocabObj.returnFilesResult())
    print(vocabObj.vocab_cv.get_feature_names())
    t1 = time.time()
    print("TOTAL TIME: " + str((t1-t0)) + "SECONDS")

if __name__=="__main__":
    main()