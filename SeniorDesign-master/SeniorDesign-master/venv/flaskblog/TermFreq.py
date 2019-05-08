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

class TermFreq:
    def __init__(self, scamFilesDirectory, notScamFilesDirectory, testFilesDirectory, directory):

        self.directory = directory
        self.scamFilesDirectory = scamFilesDirectory
        self.notScamFilesDirectory = notScamFilesDirectory
        self.testFilesDirectory = testFilesDirectory

        self.files = []
        self.testFiles = []

        self.labels = []
        self.labelIndex = defaultdict(list)

        self.trainFiles = []
        self.trainLabels = []
        self.splitTestFiles = []
        self.testLabels = []

        self.cv = CountVectorizer(stop_words='english')
        self.acc_cv = CountVectorizer(stop_words='english')

        self.termDocsTrain = []
        self.accTermDocsTrain = []
        self.termDocsTest = []
        self.accTermDocsTest = []

        self.multi = MultinomialNB(alpha=1, fit_prior=True)

        self.processed = False
        self.computed = False
        self.accuracyComputed = False
        self.fileNames = []
        self.results = []
        self.accuracy = 0.0

    def extractSingleTextFile(self, filename):
        with open(filename, 'r', encoding="utf-8") as infile:
            s = infile.read()
            return s.translate(str.maketrans('', '', string.punctuation))

    def importFilesCleanTextGenrateLabels(self):
        all_names = set(names.words())
        lemmatizer = WordNetLemmatizer()

        for filename in glob.glob(os.path.join(self.scamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            self.files.append(
                ' '.join([lemmatizer.lemmatize(word.lower())
                          for word in doc.split()
                          if word.isalpha() and word not in all_names])
            )
            self.labels.append(1)
        for filename in glob.glob(os.path.join(self.notScamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            self.files.append(
                ' '.join([lemmatizer.lemmatize(word.lower())
                          for word in doc.split()
                          if word.isalpha() and word not in all_names])
            )
            self.labels.append(0)

    def importAllTestFiles(self):
        all_names = set(names.words())
        lemmatizer = WordNetLemmatizer()

        self.testFiles = []
        for filename in glob.glob(os.path.join(self.testFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            self.testFiles.append(
                ' '.join([lemmatizer.lemmatize(word.lower())
                          for word in doc.split()
                          if word.isalpha() and word not in all_names])
            )

    def getLabelIndex(self):
        for index, label in enumerate(self.labels):
            self.labelIndex[label].append(index)

    def generateTrainingSplit(self):
        self.trainFiles, self.splitTestFiles, self.trainLabels, self.testLabels = train_test_split(
            self.files, self.labels, test_size=0.25, random_state=42)

    def generateTrainingSplitTrainedModel(self):
        self.accTermDocsTrain = self.acc_cv.fit_transform(self.trainFiles)

    def generateTrainedModel(self):
        self.termDocsTrain = self.cv.fit_transform(self.files)

    def previous(self):
        self.importFilesCleanTextGenrateLabels()
        self.getLabelIndex()
        self.processed = True

    def getResults(self):
        self.generateTrainedModel()
        self.multi.fit(self.termDocsTrain, self.labels)
        self.importAllTestFiles()
        self.termDocsTest = self.cv.transform(self.testFiles)
        self.results = self.multi.predict_proba(self.termDocsTest)
        #print(self.results)
        self.computed = True

    def getAccuracyResults(self):
        self.generateTrainingSplit()
        self.generateTrainingSplitTrainedModel()
        self.multi.fit(self.accTermDocsTrain, self.trainLabels)
        self.accTermDocsTest = self.acc_cv.transform(self.splitTestFiles)
        self.accuracy = self.multi.score(self.accTermDocsTest, self.testLabels)
        self.accuracyComputed = True