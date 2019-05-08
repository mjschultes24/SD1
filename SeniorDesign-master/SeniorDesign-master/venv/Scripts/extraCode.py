#
# icosDoc = cv.build_tokenizer()
#
# tokens = icosDoc(cleanedicos[0])

# print(tokens)
# mcprint(nltk.corpus.genesis.words('english-web.txt'))

# bigram = nltk.collocations.BigramAssocMeasures()
#
# finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
# finder2 = BigramCollocationFinder.from_words(tokens)

# listOfBigrams = list(nltk.bigrams(tokens))
#
# ## NN = noun, VBD = verb (enabled), VBN = verb (enabled), VBG = verb (crowdsourcing),
# ## adjective = JJ, RB =  adverbs
#
# for set in listOfBigrams:
#     word = nltk.pos_tag(set)
#     print(word)
    #print(nltk.pos_tag(word))
    # print(nltk.pos_tag(set[0]))
    # print(nltk.pos_tag(set[1]))

# listOfBigramsWithPOS = list(nltk.pos_tag(listOfBigrams))
#
# print(listOfBigramsWithPOS)

# cv = CountVectorizer(stop_words="english", ngram_range=(2,2))
#
# icosDoc = cv.build_tokenizer()
#
# tokens = icosDoc(cleanedicos[0])
#
# print("tokens!!")
# print(tokens)




#splitList = train_test_split(cleanedEmails, labels, test_size=0.33, random_state = 42)

#print(splitList)

# ##retrain based on the training split
#
# termDocsTrain = cv.fit_transform(splitList[0])
# labelIndex = getLabelIndex(splitList[2])
# prior = getPrior(labelIndex)
# likelihood = get_likelihood(termDocsTrain, labelIndex, smoothing)
#
#
# ## predict
# termDocsTest = cv.transform(splitList[1])
# posterior = getPosterior(termDocsTest, prior, likelihood)
#
# correct = 0.0
#
# for pred, actual in zip(posterior, splitList[3]):
#     if actual == 1:
#         if pred[1] >= 0.5:
#             correct += 1
#         elif pred[0] > 0.5:
#             correct += 1
#
# print('The accuracy on {0} testing samples is:'
#       '{1:.1f}%'.format(len(splitList[3]), correct/len(splitList[2])*100))
#