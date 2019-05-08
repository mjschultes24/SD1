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
    def __init__(self, scamFilesDirectory, notScamFilesDirectory, testFilesDirectory, vocabFile, directory):

        self.vocabList = {}
        self.directory = directory
        self.scamFilesDirectory = scamFilesDirectory
        self.notScamFilesDirectory = notScamFilesDirectory
        self.testFilesDirectory = testFilesDirectory

        self.vocabFile = vocabFile

        self.unigramList = {}
        self.bigramList = {}
        self.trigramList = {}

        self.vocabOnlyFile = []
        self.vocabOnlyTestFiles = []

        self.vocabLabels = []
        self.vocabLabelIndex = defaultdict(list)

        self.vocabTrainFiles = []
        self.vocabTrainLabels = []
        self.vocabTestFiles = []
        self.vocabTestLabels = []

        self.vocab_cv = CountVectorizer(max_features=25)
        self.vocabAcc_cv = CountVectorizer(max_features=25)

        self.vocab_termDocsTrain = []
        self.vocab_accTermDocsTrain = []
        self.vocab_termDocsTest = []
        self.vocab_accTermDocsTest = []

        self.vocabPrior = {}
        self.vocabLikelihood = {}
        self.vocabPosteriors = []

        self.multi = MultinomialNB(alpha=1, fit_prior=True)

        self.processed = False
        self.computed = False
        self.accuracyComputed = False
        self.fileNames = []
        self.vocabResults = []
        self.accuracy = 0.0

    def extractSingleTextFile(self, filename):
        with open(filename, 'r', encoding="utf-8") as infile:
            s = infile.read()
            return s.translate(str.maketrans('', '', string.punctuation))

    def importAllAndExtractVocabAndGenerateLabels(self):

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

        self.vocabOnlyTestFiles = []
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
        for index, label in enumerate(self.vocabLabels): #self.vocabOnlyFile
            self.vocabLabelIndex[label].append(index)

    def getVocabTrainingSplit(self):
        self.vocabTrainFiles, self.vocabTestFiles, self.vocabTrainLabels, self.vocabTestLabels = train_test_split(self.vocabOnlyFile, self.vocabLabels, test_size=0.25, random_state=42)

    def generateTrainingSplitTrainedModel(self):
        self.vocab_accTermDocsTrain = self.vocabAcc_cv.fit_transform(self.vocabTrainFiles)

    def generateTrainedVocabModel(self):
        self.vocab_termDocsTrain = self.vocab_cv.fit_transform(self.vocabOnlyFile)

    def vocabPreprocessingStep(self):
        self.importVocabList()
        self.importAllAndExtractVocabAndGenerateLabels()
        self.getVocabLabelIndex()
        self.processed = True

    def normalVocabTrain(self):
        self.generateTrainedVocabModel()
        self.multi.fit(self.vocab_termDocsTrain, self.vocabLabels)

    def returnVocabFilesResult(self):
        self.normalVocabTrain()
        self.importAllTestFilesAndExtractVocab()
        self.vocab_termDocsTest = self.vocab_cv.transform(self.vocabOnlyTestFiles)
        self.vocabResults = self.multi.predict_proba(self.vocab_termDocsTest)
        self.computed = True

    def getVocabAccuracy(self):
        self.getVocabTrainingSplit()
        self.generateTrainingSplitTrainedModel()
        self.multi.fit(self.vocab_accTermDocsTrain, self.vocabTrainLabels)
        self.vocab_accTermDocsTest = self.vocabAcc_cv.transform(self.vocabTestFiles)
        accuracy = self.multi.score(self.vocab_accTermDocsTest, self.vocabTestLabels)
        self.accuracy = accuracy
        self.accuracyComputed = True


################################################## CURRENTLY UNUSED METHODS ##################################################
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

    def getVocabPrior(self):
        self.vocabPrior = {label: len(index) for label, index in self.vocabLabelIndex.items()}
        totalCount = sum(self.vocabPrior.values())
        for label in self.vocabPrior:
            self.vocabPrior[label] /= float(totalCount)

    def getVocabLikelihood(self, smoothing=1):
        for label, index in self.vocabLabelIndex.items():
            ##below sums values at each row(email index) and adds 1 for smoothing - P(S,features) - all divisible by the same number
            ##it's a sparse matrix so some of the values are 0
            self.vocabLikelihood[label] = self.vocab_termDocsTrain[index, :].sum(
                axis=0) + smoothing  # sums all the values
            self.vocabLikelihood[label] = np.asarray(self.vocabLikelihood[label])[0]

            totalCount = self.vocabLikelihood[label].sum()
            self.vocabLikelihood[label] = self.vocabLikelihood[label] / float(totalCount)