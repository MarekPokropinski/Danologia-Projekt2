import json

import tweepy
import preprocessor
import pandas as pd
import numpy as np
from random import choice


def _load_credentials() -> dict:
    with open('twitter-credentials.json') as f:
        return json.load(f)[0]


def _authenticate(cred: dict) -> tweepy.API:
    auth = tweepy.OAuthHandler(cred['api-key'], cred['api-secret'])
    auth.set_access_token(cred['access-key'], cred['access-secret'])
    return tweepy.API(auth)


def get_tweets(hashtag: str, count: int, out_file: str = None):
    api = _authenticate(_load_credentials())

    res = []

    for tweet in tweepy.Cursor(api.search, q=hashtag, lang='en',
                               tweet_mode='extended', count=200).items(count):

        res.append([
            preprocessor.clean(tweet.full_text),
            tweet.created_at,
            choice([0, 1, 2])
        ])

    return res


keyword = '#ryanair'

tweets = get_tweets(keyword, 2500)

df = pd.DataFrame(tweets, columns=['text', 'date', 'sentiment'])

print(df)

df.to_csv('ryanair.csv', index=False)
