from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from spacy import displacy
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

import en_core_web_sm
import nltk
import numpy as np
import pandas as pd
import time

weights = {
    "1": 1,
    "2": 0.1,
    "3": 0.5,
    "4": 0.25,
    "else": 0.0
}


def _calculateFullScores(sentenceScores, namedEntityScores, counts):
    scaler = MinMaxScaler()
    weightList= []
    sentenceIndex = 0

    if counts[2] == 0:
        counts.pop[2]

    for i in range(len(counts)):
        for j in range(counts[i]):
            if i > 3:
                weightList.append(weights["else"])
            else:
                weightList.append(weights[str(i+1)])

    df = pd.DataFrame({
        "Weights": weightList,
        "SentenceScores": sentenceScores,
        "EntityScores": namedEntityScores,
    })

    df[["SentencesScaled"]] = scaler.fit_transform(df[["SentenceScores"]])
    df[["EntitiesScaled"]] = scaler.fit_transform(df[["EntityScores"]])
    df["S_weight"] = df["SentencesScaled"] + (2 * df["EntitiesScaled"]) + df["Weights"]

    return df["S_weight"].tolist()


def _convertHtmlToStr(elements):
    str = ""
    for element in elements:
        if len(element.text.split()) > 1:
            str += element.text
            if not str.endswith("."):
                str += ". "
    sentences = sent_tokenize(str)
    return str, len(sentences)


def _getNamedEntities(article):
    nlp = en_core_web_sm.load()
    doc = nlp(article)
    namedEntities = []
    
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "PERSON":
                namedEntities.append(ent.text)

    return namedEntities


def _getSentencesWithMaxWeights(weights, sentences):
    arr = np.array(weights)
    indexes = np.argpartition(arr, -10)[-10:]
    sentences = np.array(sentences)
    return sentences[indexes]


def _preProcess(document):
    stopwords = list(set(nltk.corpus.stopwords.words('english')))
    WN_lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(document)
    processedSentences = []
    tokens = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [WN_lemmatizer.lemmatize(word, pos="v") for word in words]

        # get rid of numbers and Stopwords
        words = [word for word in words if word.isalpha() and word not in stopwords]
        processedSentences.append(' '.join(word for word in words))
        tokens.extend(words)

    return processedSentences, tokens


def _scrapeArticle(url):
    article = ""
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)

    # Wait for article to fully load
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, 'lxml')
    strElement = ""
    countTitle, countAbstract, countH2, countH3, countH4, countP = 0, 0, 0, 0, 0, 0

    strElement, countTitle = _convertHtmlToStr(soup.find("h1", {"class": "document-title"}))
    article += strElement
    article += ". "
    strElement, countAbstract = _convertHtmlToStr(soup.find("div", {"class": "abstract-text"}))
    article += strElement

    articleHtmlBody = soup.find("div", {"id": "article"})
    if articleHtmlBody == None:
        raise ValueError

    strElement, countH2 = _convertHtmlToStr(articleHtmlBody.find_all("h2"))
    article += strElement
    strElement, countH3 = _convertHtmlToStr(articleHtmlBody.find_all("h3"))
    article += strElement
    strElement, countH4 = _convertHtmlToStr(articleHtmlBody.find_all("h4"))
    article += strElement
    strElement, countP = _convertHtmlToStr(articleHtmlBody.find_all("p"))
    article += strElement
    
    driver.close()

    counts = [countTitle, countAbstract, countH2, countH3, countH4, countP]

    return article, counts


def _tfidfScores(corpus, sentences):
    tfidf = TfidfVectorizer()
    fittedVectorizer = tfidf.fit(corpus)
    vectors = fittedVectorizer.transform(sentences).toarray()

    scores = []
    for i in range(len(vectors)):
        score = 0
        for j in range(len(vectors[i])):
            score = score + vectors[i][j]

        scores.append(score)
    return scores


def calculateSWeigth(documentType, path):
    article = ""
    if documentType == "url":
        article, counts = _scrapeArticle(path)

    sentences, tokens = _preProcess(article)
    sentenceTfidfScores = _tfidfScores(tokens, sentences)
    namedEntitiesTfidfScores = _tfidfScores(_getNamedEntities(article), sentences)
    SWeight = _calculateFullScores(sentenceTfidfScores, namedEntitiesTfidfScores, counts)
    topSentences = _getSentencesWithMaxWeights(SWeight, sent_tokenize(article))
    return topSentences

