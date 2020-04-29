import json
import tqdm

import tweepy
import preprocessor
import pandas as pd


def _load_credentials() -> dict:
    with open('twitter-credentials.json') as f:
        return json.load(f)[1]


def _authenticate(cred: dict) -> tweepy.API:
    auth = tweepy.OAuthHandler(cred['api-key'], cred['api-secret'])
    auth.set_access_token(cred['access-key'], cred['access-secret'])
    return tweepy.API(auth)


def get_tweets(hashtag: str, lang: str, count: int = 0, out_file: str = None):
    api = _authenticate(_load_credentials())

    res = []

    try:
        for tweet in tqdm.tqdm(tweepy.Cursor(api.search, q=f'#{hashtag}', lang=lang,
                                             tweet_mode='extended', count=100).items(count)):
            res.append([
                preprocessor.clean(tweet.full_text),
                tweet.created_at,
                -1
            ])
    except:
        pass

    df = pd.DataFrame(res, columns=['text', 'date', 'sentiment'])

    return df

# keyword = 'ryanair'

# tweets = get_tweets(keyword)

# tweets.to_csv(f'{keyword}.csv', index=False)
