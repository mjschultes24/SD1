
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
    #os.chdir(directory)
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

    correct = 0.0
    correctPrediction = "correct prediction"
    incorrectPrediction = "incorrect prediction"

    for pred, actual in zip(posterior, yTest):
        if actual == 1:
            if pred[1] >= 0.5:
                correct += 1
                print(correctPrediction + "actual: " + str(actual) + " rounded prediction " + str(1) + ", actual prediction " + str(pred[1]))
            else:
                print(incorrectPrediction + "actual: " + str(actual) + " prediction " + str(0) + ", actual prediction " + str(pred[0]))
        else:
            if pred[0] > 0.5:
                correct += 1
                print(correctPrediction + "actual: " + str(actual) + " prediction " + str(0) + ", actual prediction " + str(pred[0]))
            else:
                print(incorrectPrediction + "actual: " + str(actual) + " prediction " + str(1) + ", actual prediction " + str(pred[1]))
    print('The accuracy on {0} testing samples is:'
          '{1:.1f}%'.format(len(yTest), correct / len(yTest) * 100))
