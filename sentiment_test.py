# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:22:25 2022

@author: Thomas
"""

import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

#Baseline sentiment scorer
sia = SentimentIntensityAnalyzer()
#test = sia.polarity_scores("Just the worst.  Really, truly terrible.  I hate this.")
test_list = ["Just the worst.","Really, truly terrible.","I hate this."]

#Get sentiment lexicon



#Import data
import pandas as pd
df = pd.read_csv('C:\\File\\Active_Dataset\\IMDB.csv')
df = df.iloc[0:100,]
colName = 'review'

def text_to_score(df,colName):
    df['sentiment_scores'] = df.apply(lambda row: sia.polarity_scores(row[colName]),axis=1)
    df['pos'] = df.apply(lambda row: row.sentiment_scores['pos'],axis=1)
    df['neu'] = df.apply(lambda row: row.sentiment_scores['neu'],axis=1)
    df['neg'] = df.apply(lambda row: row.sentiment_scores['neg'],axis=1)
    df['compound'] = df.apply(lambda row: row.sentiment_scores['compound'],axis=1)
    df = df.drop(['sentiment_scores'],axis=1)
    return(df)

def str_list_to_score_list(lst):
    return([sia.polarity_scores(x) for x in lst])

#Sentence by sentence scoring
def text_to_score_list(df,colName):
    df['sentence_list'] = df.apply(lambda row: nltk.tokenize.sent_tokenize(row[colName]),axis=1)
    df['score_list'] = df.apply(lambda row: str_list_to_score_list(row[colName]),axis=1)



df = text_to_score(df,'review')
    #Format into list of entries w/ label
    #Tokenize
    #Match adjectives to nouns
    #Apply Sentiment to nouns based on adjectives
        #Score entries
        #Count of positive/negative/neutral
        #Location of positive/negative/neutral
#Filter out positive/neutral/full negative entries
#Examine data for matches
#Examine matches for relative location
#Develop method for identifying matches