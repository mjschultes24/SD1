# classification.py
# M. Schultes
# 4/7/2019

# Input: filepath
# Output: spreadsheet

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
from nltk.util import ngrams

import nltk
import glob
import os
import string
import xlwt


def Load_Text_Files(file_path):
    whitepapers = []
    for filename in glob.glob(os.path.join(file_path, '*.txt')):
        with open(filename, 'r', encoding="ISO-8859-1") as infile:
            s = infile.read()
            # Removes punctuation so only raw text
            whitepapers.append(s.translate(
                str.maketrans('', '', string.punctuation)))
    return whitepapers


def Letters_Only(astr):
    return astr.isalpha()


def Remove_Unicode(docs):
    docs_with_no_unicode = []
    for doc in docs:
        filtered_word = ''
        filtered_word = ''.join(
            [x if x in string.printable else ' ' for x in doc])
        docs_with_no_unicode.append(filtered_word)
    return docs_with_no_unicode


def Clean_Text(docs):
    cleaned_docs = []
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                      for word in doc.split()
                                      if Letters_Only(word)
                                      and word not in all_names
                                      and len(word) > 2]))
    return cleaned_docs


def Tokenize_Text(docs):
    stopWords = set(stopwords.words('english'))
    tokenized_docs = []
    tokens = []
    for doc in docs:
        tokenized_docs = word_tokenize(doc)
        for token in tokenized_docs:
            if token not in stopWords:
                tokens.append(token)
    return tokens


def Dictionary_to_List(docs):
    keys = []
    values = []
    for k, v in docs.items():
        if v > 2:
            keys.append(str(k))
            values.append(v)
    return keys, values


def Remove_Duplicate(list1, freq1, list2, freq2):
    not_duplicates = []
    not_duplicates_freq = []
    match_variable = 0
    for x in list1:
        for y in list2:
            if x == y:
                match_variable = 1
        if match_variable == 0:
            not_duplicates.append(x)
            not_duplicates_freq.append(freq1[list1.index(x)])
        match_variable = 0
    return not_duplicates, not_duplicates_freq


def Output(filename, sheet, list1, list2):
    book = xlwt.Workbook()
    sh = book.add_sheet(sheet)

    col1_name = 'Terms'
    col2_name = 'Frequencies'

    n = 0

    sh.write(n, 0, col1_name)
    sh.write(n, 1, col2_name)

    for m, e1 in enumerate(list1, n+1):
        sh.write(m, 0, e1)

    for m, e2 in enumerate(list2, n+1):
        sh.write(m, 1, e2)

    book.save(filename)


def main():
    print("Program Init.")

    # Load in raw text (could prompt user here for filepath)
    whitepapers_raw_text = []
    file_path = "C:/Users/michael/Documents/NotScamRawText"
    whitepapers_raw_text = Load_Text_Files(file_path)

    scam_whitepapers_raw_text = []
    scam_file_path = "C:/Users/michael/Documents/ScamRawText"
    scam_whitepapers_raw_text = Load_Text_Files(scam_file_path)

    # Clean Text
    # - Remove non-alpha characters
    # - Strip non-printable characters
    # - Remove names
    # - Remove words less than 2 characters long
    # - Lemmantize (sort words by grouping inflected or variant forms of the same word)
    whitepapers_with_no_unicode = []
    whitepapers_with_no_unicode = Remove_Unicode(whitepapers_raw_text)

    whitepapers_cleaned = []
    whitepapers_cleaned = Clean_Text(whitepapers_with_no_unicode)

    scam_whitepapers_with_no_unicode = []
    scam_whitepapers_with_no_unicode = Remove_Unicode(scam_whitepapers_raw_text)

    scam_whitepapers_cleaned = []
    scam_whitepapers_cleaned = Clean_Text(scam_whitepapers_with_no_unicode)

    # - Remove stopwords
    # - Tokenize text
    whitepapers_tokens = []
    whitepapers_tokens = Tokenize_Text(whitepapers_cleaned)

    scam_whitepapers_tokens = []
    scam_whitepapers_tokens = Tokenize_Text(scam_whitepapers_cleaned)

    # Group words into ngrams
    unigrams = nltk.FreqDist(whitepapers_tokens)
    bigrams = Counter(ngrams(whitepapers_tokens, 2))
    trigrams = Counter(ngrams(whitepapers_tokens, 3))
    fourgrams = Counter(ngrams(whitepapers_tokens, 4))
    fivegrams = Counter(ngrams(whitepapers_tokens, 5))

    scam_unigrams = nltk.FreqDist(scam_whitepapers_tokens)
    scam_bigrams = Counter(ngrams(scam_whitepapers_tokens, 2))
    scam_trigrams = Counter(ngrams(scam_whitepapers_tokens, 3))
    scam_fourgrams = Counter(ngrams(scam_whitepapers_tokens, 4))
    scam_fivegrams = Counter(ngrams(scam_whitepapers_tokens, 5))

    unigrams_terms = []
    unigrams_frequency = []
    unigrams_terms, unigrams_frequency = Dictionary_to_List(unigrams)

    bigrams_terms = []
    bigrams_frequency = []
    bigrams_terms, bigrams_frequency = Dictionary_to_List(bigrams)

    trigrams_terms = []
    trigrams_frequency = []
    trigrams_terms, trigrams_frequency = Dictionary_to_List(trigrams)

    fourgrams_terms = []
    fourgrams_frequency = []
    fourgrams_terms, fourgrams_frequency = Dictionary_to_List(fourgrams)

    fivegrams_terms = []
    fivegrams_frequency = []
    fivegrams_terms, fivegrams_frequency = Dictionary_to_List(fivegrams)

    scam_unigrams_terms = []
    scam_unigrams_frequency = []
    scam_unigrams_terms, scam_unigrams_frequency = Dictionary_to_List(scam_unigrams)

    scam_bigrams_terms = []
    scam_bigrams_frequency = []
    scam_bigrams_terms, scam_bigrams_frequency = Dictionary_to_List(scam_bigrams)

    scam_trigrams_terms = []
    scam_trigrams_frequency = []
    scam_trigrams_terms, scam_trigrams_frequency = Dictionary_to_List(scam_trigrams)

    scam_fourgrams_terms = []
    scam_fourgrams_frequency = []
    scam_fourgrams_terms, scam_fourgrams_frequency = Dictionary_to_List(scam_fourgrams)

    scam_fivegrams_terms = []
    scam_fivegrams_frequency = []
    scam_fivegrams_terms, scam_fivegrams_frequency = Dictionary_to_List(scam_fivegrams)

    # Subtract lists
    only_scam_unigrams = []
    only_scam_unigrams_frequencies = []
    only_scam_unigrams, only_scam_unigrams_frequencies = Remove_Duplicate(scam_unigrams_terms, scam_unigrams_frequency, unigrams_terms, unigrams_frequency)

    only_scam_bigrams = []
    only_scam_bigrams_frequencies = []
    only_scam_bigrams, only_scam_bigrams_frequencies = Remove_Duplicate(scam_bigrams_terms, scam_bigrams_frequency, bigrams_terms, bigrams_frequency)

    only_scam_trigrams = []
    only_scam_trigrams_frequencies = []
    only_scam_trigrams, only_scam_trigrams_frequencies = Remove_Duplicate(scam_trigrams_terms, scam_trigrams_frequency, trigrams_terms, trigrams_frequency)

    only_scam_fourgrams = []
    only_scam_fourgrams_frequencies = []
    only_scam_fourgrams, only_scam_fourgrams_frequencies = Remove_Duplicate(scam_fourgrams_terms, scam_fourgrams_frequency, fourgrams_terms, fourgrams_frequency)

    only_scam_fivegrams = []
    only_scam_fivegrams_frequencies = []
    only_scam_fivegrams, only_scam_fivegrams_frequencies = Remove_Duplicate(scam_fivegrams_terms, scam_fivegrams_frequency, fivegrams_terms, fivegrams_frequency)

    Output('subtracted_unigrams.xls', 'Sheet 1', only_scam_unigrams, only_scam_unigrams_frequencies)
    Output('subtracted_bigrams.xls', 'Sheet 1', only_scam_bigrams, only_scam_bigrams_frequencies)
    Output('subtracted_trigrams.xls', 'Sheet 1', only_scam_trigrams, only_scam_trigrams_frequencies)
    Output('subtracted_fourgrams.xls', 'Sheet 1', only_scam_fourgrams, only_scam_fourgrams_frequencies)
    Output('subtracted_fivegrams.xls', 'Sheet 1', only_scam_fivegrams, only_scam_fivegrams_frequencies)

    Output('scam_unigrams.xls', 'Sheet 1', scam_unigrams_terms, scam_unigrams_frequency)
    Output('scam_bigrams.xls', 'Sheet 1', scam_bigrams_terms, scam_bigrams_frequency)
    Output('scam_trigrams.xls', 'Sheet 1', scam_trigrams_terms, scam_trigrams_frequency)
    Output('scam_fourgrams.xls', 'Sheet 1', scam_fourgrams_terms, scam_fourgrams_frequency)
    Output('scam_fivegrams.xls', 'Sheet 1', scam_fivegrams_terms, scam_fivegrams_frequency)

    print("Program Comp.")


if __name__ == "__main__":
    main()
