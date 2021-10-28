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

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import en_core_web_sm
import nltk
import numpy as np
import time


def _calculateFullScores(sentenceScores, namedEntityScores):
    scaler = MinMaxScaler()
    df = pd.DataFrame({
        "SentenceScores": sentenceScores,
        "EntityScores": namedEntityScores
    })
    df[["SentencesScaled"]] = scaler.fit_transform(df[["SentenceScores"]])
    df[["EntitiesScaled"]] = scaler.fit_transform(df[["EntityScores"]])
    df["S_weight"] = df["SentencesScaled"] + 2 * df["EntitiesScaled"]

    return df["S_weight"].tolist()


def _convertHtmlToStr(elements):
    str = ""
    for element in elements:
        if len(element.text.split()) > 1:
            str += element.text
            if not str.endswith("."):
                str += ". "
    return str


def _getNamedEntities(article):
    nlp = en_core_web_sm.load()
    doc = nlp(article)
    namedEntities = []
    
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "PERSON":
                namedEntities.append(ent.text)

    return namedEntities



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

    article += soup.find("h1", {"class": "document-title"}).text
    article += ". "
    article +=  soup.find("div", {"class": "abstract-text"}).text
    article += " "

    articleHtmlBody = soup.find("div", {"id": "article"})
    if articleHtmlBody == None:
        raise ValueError

    article += _convertHtmlToStr(articleHtmlBody.find_all("h2"))
    article += _convertHtmlToStr(articleHtmlBody.find_all("h3"))
    article += _convertHtmlToStr(articleHtmlBody.find_all("h4"))
    article += _convertHtmlToStr(articleHtmlBody.find_all("p"))
    
    driver.close()

    return article


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
        article = _scrapeArticle(path)

    sentences, tokens = _preProcess(article)
    sentenceTfidfScores = _tfidfScores(tokens, sentences)
    namedEntitiesTfidfScores = _tfidfScores(_getNamedEntities(article), sentences)
    SWeight = _calculateFullScores(sentenceTfidfScores, namedEntitiesTfidfScores)
    print(SWeight)

