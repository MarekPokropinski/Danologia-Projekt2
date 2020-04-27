import tensorflow as tf
import numpy as np
from model import build_model
import dataset
import twitter

import os
# turn of gpu acceleration if we don't need it
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load tokenizer and model weights from files created by notebook
with open('tokenizer.json', 'r') as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
model = build_model(tokenizer.num_words)
model.load_weights('best_model.h5')


def predict(texts):
    '''
    Takes list of texts (strings) to run predictions on.
    returns np.array of shape (len(texts),) with predicted sentiments
    0 - negative
    1 - neutral
    2 - positive
    '''
    sequences = tokenizer.texts_to_sequences(texts)
    data = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, padding='post')
    return np.argmax(model.predict(data, batch_size=128), axis=1)


def predict_for_hashtag(hashtag, lang='en'):
    tweets = twitter.get_tweets(hashtag, lang)
    tweets['sentiment'] = predict(tweets['text'])
    return tweets


if __name__ == '__main__':
    print(predict(["I hate you all",
                   "You know who helps me almost everyday?",
                   "Hello guys, i love you :)"
                   ]))
