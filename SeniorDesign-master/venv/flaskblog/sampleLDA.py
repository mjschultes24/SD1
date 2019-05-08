import pandas as pd
import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim import corpora, models
from nltk.stem.porter import *
from pprint import pprint
from gensim.test.utils import common_corpus
from sklearn.model_selection import train_test_split
from nltk.corpus import names, wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer

import time
import numpy as np
import nltk
import string
import glob
import cryptoAPIs


class LDAModel:

    def __init__(self, scamFilesDirectory, notScamFilesDirectory, testFilesDirectory, directory):
        self.directory = directory
        self.scamFilesDirectory = scamFilesDirectory
        self.notScamFilesDirectory = notScamFilesDirectory
        self.testFilesDirectory = testFilesDirectory
        self.directory = directory

        self.files = []
        self.scamFiles = []
        self.testFiles = []

        self.trainFiles = []
        self.testFiles = []

        self.dictionary = None
        self.bow = []
        self.tfidfModel = None
        self.tfidfTransformed = None
        self.ldaModel = None
        self.topics = []

        self.dictionaryScam = None
        self.bowScam = []
        self.tfidfModelScam = None
        self.tfidfTransformedScam = None
        self.ldaModelScam = None
        self.topicsScam = []

        self.preprocess = False
        self.compute = False

    def extractSingleTextFile(self, filename):
        with open(filename, 'r', encoding="utf-8") as infile:
            s = infile.read()
            return s.translate(str.maketrans('', '', string.punctuation))

    def importFilesCleanTextGenrateLabels(self):
        all_names = set(names.words())
        lemmatizer = WordNetLemmatizer()
        os.chdir(self.directory)

        for filename in glob.glob(os.path.join(self.scamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            fileString = (
                ' '.join([lemmatizer.lemmatize(word.lower())
                          for word in doc.split()
                          if word.isalpha() and word not in all_names])
            )
            self.scamFiles.append(nltk.word_tokenize(fileString))

        for filename in glob.glob(os.path.join(self.notScamFilesDirectory, '*.txt')):
            doc = self.extractSingleTextFile(filename)
            fileString = (
                ' '.join([lemmatizer.lemmatize(word.lower())
                          for word in doc.split()
                          if word.isalpha() and word not in all_names])
            )
            self.files.append(nltk.word_tokenize(fileString))

    def tokenizeFiles(self):
        for doc in self.scamFiles:
            nltk.word_tokenize(doc)
        for doc in self.files:
            nltk.word_tokenize(doc)

    def fillDictionary(self):
        self.dictionaryScam = gensim.corpora.Dictionary(self.scamFiles)
        self.dictionaryScam.filter_extremes(no_below = 5)

        self.dictionary = gensim.corpora.Dictionary(self.files)
        self.dictionary.filter_extremes(no_below=5)

    def fillBagOfWords(self):
        for text in self.scamFiles:
            self.bowScam.append(self.dictionaryScam.doc2bow(text))

        for text in self.files:
            self.bow.append(self.dictionary.doc2bow(text))

    def fitTfIDFModel(self):
        self.tfidfModel = models.TfidfModel(self.bow)
        self.tfidfTransformed = self.tfidfModel[self.bow]

        self.tfidfModelScam = models.TfidfModel(self.bowScam)
        self.tfidfTransformedScam = self.tfidfModel[self.bowScam]

    def generateLdaModel(self):
        self.ldaModel = gensim.models.LdaModel(self.tfidfTransformed, num_topics=10,
                                               id2word=self.dictionary, passes=100)

        self.ldaModelScam = gensim.models.LdaModel(self.tfidfTransformedScam, num_topics=10,
                                               id2word=self.dictionaryScam, passes=100)

    def printTopics(self):
        print(self.ldaModel.show_topics())

    def previous(self):
        self.importFilesCleanTextGenrateLabels()
        self.preprocess = True

    def getTopicWordsFromModel(self):
        topics = self.ldaModel.print_topics(num_topics=10,
                                    num_words=10)
        for list in topics:
            setList = []
            for topicWord in list[1].split(" + "):
                m = re.search(r'([a-zA-Z]+)', topicWord)
                if m is not None:
                    setList.append(m.group())
            self.topics.append(setList)

        topicsScam = self.ldaModelScam.print_topics(num_topics=10,
                                            num_words=10)
        for list in topicsScam:
            setList = []
            for topicWord in list[1].split(" + "):
                m = re.search(r'([a-zA-Z]+)', topicWord)
                if m is not None:
                    setList.append(m.group())
            self.topicsScam.append(setList)

    def generateTopics(self):
        self.fillDictionary()
        self.fillBagOfWords()
        self.fitTfIDFModel()
        self.generateLdaModel()
        self.compute = True
