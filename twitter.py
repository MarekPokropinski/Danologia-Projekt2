import json

import preprocessor as p
import tweepy

keyword = '#ryanair'

with open('twitter-credentials.json') as f:
    cred = json.load(f)

auth = tweepy.OAuthHandler(cred['api-key'], cred['api-secret'])
auth.set_access_token(cred['access-key'], cred['access-secret'])
api = tweepy.API(auth)

for idx, tweet in enumerate(tweepy.Cursor(api.search, q=keyword, lang='en',
                                          tweet_mode='extended',
                                          count=200).items()):
    print(idx)
    print(p.clean(tweet.full_text))
    # print(tweet.text)
    # print(tweet.truncated)
