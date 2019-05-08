from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

import glob
import os
import numpy as np
import string
import collections
import xlwt

# Initialize data structures, labels refers to classifcation of 'potential scam' (0) or 'not potential scam' (1)
whitepapers, labels = [], []

# Load in the raw txt files and remove punctuation, affix labels
def load_text_files(file_path, label):
        for filename in glob.glob(os.path.join(file_path, '*.txt')):
                with open(filename, 'r', encoding = "ISO-8859-1") as infile:
                        s=infile.read()
                        whitepapers.append(s.translate(str.maketrans('', '', string.punctuation)))
                        labels.append(label)

file_path = "C:/Users/michael/Documents/ScamRawText/"
load_text_files(file_path, 0)

file_path = "C:/Users/michael/Documents/NotScamRawText/"
load_text_files(file_path, 1)

# Remove numbers and other non-alpha characters
def letters_only(astr):
    return astr.isalpha()

whitepapers_with_no_unicode = []
for whitepaper in whitepapers:
    filtered_word = ''
    filtered_word = ''.join([x if x in string.printable else ' ' for x in whitepaper])
    whitepapers_with_no_unicode.append(filtered_word)

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

# Remove names and lemmatize (sort words by grouping inflected or variant forms of the same word)
def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                        for word in doc.split()
                                        if letters_only(word)
                                        and word not in all_names
                                        and len(word) > 2]))
    return cleaned_docs

cleaned_whitepapers = clean_text(whitepapers_with_no_unicode)

# Tokenize the text and stem (reducing inflected/derived words to their word stem)
def process_text(text, stem=False):
    tokens = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# Transform texts to Tf-Idf (term frequency–inverse document frequency) coordinates
def cluster_texts(texts, clusters):
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 lowercase=True,
                                 max_df=0.49,
                                 min_df=0.02,
                                 max_features=500,
                                 ngram_range=(1,5),
                                 stop_words="english") 
 
    # Transform texts to Tf-Idf (term frequency–inverse document frequency) coordinates
    tfidf_model = vectorizer.fit_transform(texts)

    # Cluster using k-means
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)

    # Output data obtained from the process
    feature_names = vectorizer.get_feature_names() # all words
    feature_mappings = vectorizer.vocabulary_ # words and the
    feature_stopwords = vectorizer.stop_words_
    feature_idfs = vectorizer.idf_

    terms = []
    tf_idfs = []

    feature_counter = 0
    f= open("feature_names.txt","w+")
    for name in feature_names:
        feature_frequency = tfidf_model.toarray().sum(axis=0)
        f.write(str(name) + " " + str(feature_frequency[feature_counter]) + "\n")
        terms.append(str(name))
        tf_idfs.append(feature_frequency[feature_counter])
        feature_counter = feature_counter + 1

    f= open("feature_mappings.txt","w+")
    for mapping in feature_mappings:
        mapping_variable = str(mapping)  
        f.write(mapping_variable + " " + str(feature_mappings[mapping_variable]) + "\n")
 
    f= open("feature_stopwords.txt","w+")
    for stopword in feature_stopwords:
        f.write(str(stopword) + "\n")

    f= open("feature_idfs.txt","w+")
    for idfs in feature_idfs:
        f.write(str(idfs) + "\n")

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    def output(filename, sheet, list1, list2):
        book = xlwt.Workbook()
        sh = book.add_sheet(sheet)

        col1_name = 'Terms'
        col2_name = 'tf_idfs'

        n = 0

        sh.write(n, 0, col1_name)
        sh.write(n, 1, col2_name)

        for m, e1 in enumerate(list1, n+1):
            sh.write(m, 0, e1)

        for m, e2 in enumerate(list2, n+1):
            sh.write(m, 1, e2)

        book.save(filename)

    output('tf-idfs.xls', 'Sheet 1', terms, tf_idfs)
 
    return clustering
 
# Main()
if __name__ == "__main__":
    print("Program Init.")

    articles = cleaned_whitepapers
    clusters = cluster_texts(articles, 2)
    pprint(dict(clusters))

    print("Program Complete.")