import re
from bs4 import BeautifulSoup
import requests
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt

baselink = 'https://www.trustpilot.com'
firstlink = 'https://www.trustpilot.com/review/olacabs.com'

def getLinks(link):
    links = []
    while link:
        links.append(link)
        req = requests.get(link)
        soup = BeautifulSoup(req.text,'html.parser')
        nextlink = soup.find('a', {'class':'next-page'})
        if nextlink:
            newlink = nextlink.get('href')
            newlink = baselink+newlink
            link = newlink
        else:
            break
    return links

def getSoups(links):
    SoupList = []
    for li in links:
        req = requests.get(li)
        SoupList.append(BeautifulSoup(req.text, 'html.parser'))
    return SoupList

def getSections(SoupList):
    sectionList = []
    for itersoup in SoupList:
        sections = itersoup.find_all('section', {'class':'review__content'})
        for section in sections:
            sectionList.append(section)
    return sectionList

def getElements(sectionList):
    headings = []
    reviews = []
    stars = []
    for section in sectionList:
        heading = section.find('h2', {'class':'review-content__title'})
        if heading:
            heading = heading.get_text()
            heading = re.sub('\n', '', heading)
        content = section.find('p', {'class':'review-content__text'})
        if content:
            content = content.get_text()
            content = re.sub('\n','', content)
        star = section.find('div', {'class':'star-rating'})
        if star:
            star = star.img['src']
            star = int(star[-5:-4])
        headings.append(heading)
        reviews.append(content)
        stars.append(star)
    return headings, reviews, stars

def makeDataFrame(headings, reviews, stars):
    df = pd.DataFrame(list(zip(headings, reviews, stars)), columns = ['Headings','Content', 'Stars'])
    return df

def getClassifier():
    classifier = TextClassifier.load('en-sentiment')
    return classifier

def getColumns(classifier, df):
    heading_labels = []
    heading_labscore = []
    content_labels = []
    content_labscore = []
    for index, row in df.iterrows():
        if row['Headings']:
            sentence = Sentence(row['Headings'])
            classifier.predict(sentence)
            heading_labels.append(sentence.labels[0].to_dict()['value'])
            heading_labscore.append(sentence.labels[0].to_dict()['confidence'])
        else:
            heading_labels.append(None)
            heading_labscore.append(None)
        if row['Content']:
            sentence = Sentence(row['Content'])
            classifier.predict(sentence)
            content_labels.append(sentence.labels[0].to_dict()['value'])
            content_labscore.append(sentence.labels[0].to_dict()['confidence'])
        else:
            content_labels.append(None)
            content_labscore.append(None)
    return heading_labels, heading_labscore, content_labels, content_labscore

def appendColumns(df, heading_labels, heading_labscore, content_labels, content_labscore):
    df['headingSentiment'] = pd.Series(heading_labels)
    df['headingSentimentScore'] = pd.Series(heading_labscore)
    df['contentSentiment'] = pd.Series(content_labels)
    df['contentSentimentScore'] = pd.Series(content_labscore)
    return df

def makeWordCloud(series):
  text = " ".join(review for review in series.astype(str))
  stopwords = set(STOPWORDS)
  wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
  plt.axis("off")
  plt.figure( figsize=(40,20))
  plt.tight_layout(pad=0)
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.show()

def Analyze(df):
  print("No. of reviews = " + str(df.shape[0]))
  print("No. of reviews with heading confidence > 50% = " + str(df[df['headingSentimentScore'] > 0.5]['headingSentimentScore'].count()))
  print("No. of reviews with content confidence > 50% = " + str(df[df['contentSentimentScore'] > 0.5]['contentSentimentScore'].count()))
  print("No. of reviews with positive headline = " + str(df[df['headingSentiment'] == 'POSITIVE']['headingSentiment'].count()))
  print("No. of reviews with negative headline = "+ str(df[df['headingSentiment'] == 'NEGATIVE']['headingSentiment'].count()))
  print("No. of reviews with positive content = " + str(df[df['contentSentiment'] == 'POSITIVE']['contentSentiment'].count()))
  print("No. of reviews with negative content = "+ str(df[df['contentSentiment'] == 'NEGATIVE']['contentSentiment'].count()))
  print("No. of NULL reviews = " + str(df['Content'].isnull().sum(axis=0)))
  print("Average no. of stars = " + "{:.2f}".format(df['Stars'].mean()) + "/5")
  print("%age of negative headlines = " + "{:.2f}".format(100*df[df['headingSentiment'] == 'NEGATIVE']['headingSentiment'].count()/df.shape[0]))
  print("%age of negative reviews = " + "{:.2f}".format(100*df[df['contentSentiment'] == 'NEGATIVE']['contentSentiment'].count()/df.shape[0]))
  print("%age of positive headlines = " + "{:.2f}".format(100*df[df['headingSentiment'] == 'POSITIVE']['headingSentiment'].count()/df.shape[0]))
  print("%age of positive reviews = " + "{:.2f}".format(100*df[df['contentSentiment'] == 'POSITIVE']['contentSentiment'].count()/df.shape[0]))

if __name__ == "main":
    link = 'https://www.trustpilot.com/review/olacabs.com'
    links = getLinks(link)
    soupList = getSoups(links)
    sectionList = getSections(soupList)
    headings, reviews, stars = getElements(sectionList)
    df = makeDataFrame(headings, reviews, stars)
    classifier = getClassifier()
    heading_labels, heading_labscore, content_labels, content_labscore = getColumns(classifier, df)
    df = appendColumns(df, heading_labels, heading_labscore, content_labels, content_labscore)
    makeWordCloud(df['Headings'])
    makeWordCloud(df['Content'])
    Analyze(df)