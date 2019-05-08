from flask import render_template, url_for, flash, redirect, request, send_from_directory
from werkzeug.utils import secure_filename
from flaskblog import app, APP_ROOT
import flaskblog.cryptoAPIs as crypto
import flaskblog.Vocab as voc
import flaskblog.TermFreq as tf
import flaskblog.sampleLDA as lda
import os
import time

testFolder = os.path.join(APP_ROOT, 'testingFiles/')
if not os.path.isdir(testFolder):
    os.mkdir(testFolder)

scamFolder = os.path.join(APP_ROOT, 'scamFiles/')
if not os.path.isdir(scamFolder):
    os.mkdir(scamFolder)

notScamFolder = os.path.join(APP_ROOT, 'notScamFiles/')
if not os.path.isdir(notScamFolder):
    os.mkdir(notScamFolder)

scamFilesDirectory = scamFolder
notScamFilesDirectory = notScamFolder
testFilesDirectory = testFolder
vocabDirectory = os.path.join(APP_ROOT, 'vocabList/vocab.xlsx')

vocabObj = voc.Vocab(scamFilesDirectory,notScamFilesDirectory,testFilesDirectory,vocabDirectory, r"C:\Users\mpalu\PycharmProjects\practiceML")
termFreqObj = tf.TermFreq(scamFilesDirectory,notScamFilesDirectory,testFilesDirectory, r"C:\Users\mpalu\PycharmProjects\practiceML")
ldaObj = lda.LDAModel(scamFilesDirectory,notScamFilesDirectory,testFilesDirectory, r"C:\Users\mpalu\PycharmProjects\practiceML")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def index():
    for file in request.files.getlist("testingFile"):
        filename = file.filename
        destination = "/".join([testFolder, filename])
        file.save(destination)
        vocabObj.computed = False
        termFreqObj.computed = False
        vocabObj.fileNames.append(file.filename)
    vocabObj.fileNames = os.listdir(testFolder)
    return render_template('index.html', title='')

@app.route("/results", methods=['GET', 'POST'])
def results():
    if not vocabObj.processed:
        vocabObj.vocabPreprocessingStep()
    if not vocabObj.computed:
        vocabObj.returnVocabFilesResult()
    if not vocabObj.accuracyComputed:
         vocabObj.getVocabAccuracy()
    ###
    if not termFreqObj.processed:
        termFreqObj.previous()
    if not termFreqObj.computed:
        termFreqObj.getResults()
    if not termFreqObj.accuracyComputed:
        termFreqObj.getAccuracyResults()
    ###
    if not ldaObj.preprocess:
        ldaObj.previous()
    if not ldaObj.compute:
        ldaObj.generateTopics()
    ldaObj.getTopicWordsFromModel()
    ran = range(len(termFreqObj.results))
    print("PRINTING NAMES")
    print(vocabObj.fileNames)
    return render_template('results.html', title='', regResult=termFreqObj.results, ran=ran, result = vocabObj.vocabResults,
                           names=vocabObj.fileNames, vocabAcc=vocabObj.accuracy, topics=ldaObj.topics, termFreqAcc=termFreqObj.accuracy,
                           topicsScam = ldaObj.topicsScam)



@app.route("/about")
def about():
    return render_template('about.html', title='')