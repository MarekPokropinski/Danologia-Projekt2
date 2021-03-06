import json
import os

import tqdm

import tweepy
import preprocessor
import pandas as pd


def _load_credentials() -> dict:
    with open('twitter-credentials.json') as f:
        return json.load(f)[0]


cache = set()


def _authenticate(cred: dict) -> tweepy.API:
    auth = tweepy.OAuthHandler(cred['api-key'], cred['api-secret'])
    auth.set_access_token(cred['access-key'], cred['access-secret'])
    return tweepy.API(auth)


def get_tweets(hashtag: str, lang: str, count: int = 0, out_file: str = None):
    if hashtag in cache:
        print(f'loaded {hashtag} from cache')
        df = pd.read_csv(f'cache/{hashtag}.csv')
        df = df.astype({'date': 'datetime64[ns]'})
        return df

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
    df.to_csv(f'cache/{hashtag}.csv')
    cache.add(hashtag)

    return df


# load cache
for file in os.listdir('cache'):
    if file.endswith('.csv'):
        print(f'found {os.path.splitext(file)[0]} in cache')
        cache.add(os.path.splitext(file)[0])

# keyword = 'ryanair'

# tweets = get_tweets(keyword)

# tweets.to_csv(f'{keyword}.csv', index=False)
