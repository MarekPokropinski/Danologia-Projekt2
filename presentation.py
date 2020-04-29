import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import ipywidgets as widgets
from predictor import predict_for_hashtag
from IPython.display import clear_output

import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random


def sentiment_to_desc(sentiment):
    if sentiment == 0:
        return 'negative'
    elif sentiment == 1:
        return 'neutral'
    else:
        return 'positive'


import numpy as np


def plot_over_time(x, y, y_titles, title):
    fig = go.Figure()
    for y_i, y_title in zip(y, y_titles):
        fig.add_trace(go.Scatter(x=x, y=y_i, name=y_title))

    fig.update_layout(
        title_text=title
    )
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.show()


def plot_mean_sentiment(df, tag):
    mean = df.groupby(['hour']).mean()
    mean = mean.reset_index()
    plot_over_time(mean.hour, [mean.sentiment], ['Mean sentiment'],
                   f"Mean sentiment of tweets with tag: {tag}")


def plot_num(df, tag):
    count_all = df.groupby(['hour']).count()
    count_all = count_all.reset_index()

    count = df.groupby(['hour', 'sentiment_desc']).count()
    count = count.reset_index()
    count[count.sentiment_desc == 'negative']

    negatives, neutrals, positives = [], [], []

    def get(df, hour, sentiment_desc):
        data = df[(df.hour == hour) & (df.sentiment_desc == sentiment_desc)]
        if len(data) == 0:
            return 0
        else:
            return int(data.text)

    for hour in sorted(count.hour.unique()):
        negatives.append(get(count, hour, 'negative'))
        neutrals.append(get(count, hour, 'neutral'))
        positives.append(get(count, hour, 'positive'))

    all_ = np.array(count_all.sentiment)
    negatives = np.array(negatives)
    neutrals = np.array(neutrals)
    positives = np.array(positives)

    plot_over_time(count_all.hour,
                   [count_all.sentiment, negatives, neutrals, positives],
                   ['all', 'negatives', 'neutrals', 'positives'],
                   f"Number of tweets with tag: {tag}")

    neg_ratio = negatives / all_
    neutr_ratio = neutrals / all_
    pos_ratio = positives / all_

    #     plot_over_time(count_all.hour,
    #                    [neg_ratio, neutr_ratio, pos_ratio],
    #                    ['negatives_ratio', 'neutrals_ratio', 'positives_ratio'],
    #                    f"Ratio of tweets with tag: {tag}")

    fig = go.Figure()
    for y_i, y_title in zip([neg_ratio, neutr_ratio, pos_ratio],
                            ['negatives', 'neutrals', 'positives']):
        fig.add_trace(
            go.Scatter(x=count_all.hour, y=y_i, name=y_title, hoverinfo='x+y', stackgroup='one'))

    fig.update_layout(
        title_text=f"Ratio of tweets with tag: {tag}"
    )
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.show()


def commons_dict(texts):
    dict = {}
    for text in texts:
        for word in text.lower().split(' '):
            if word in dict:
                dict[word] = dict[word] + 1
            else:
                dict[word] = 1

    summ = sum([dict[key] for key in dict])
    for key in dict:
        dict[key] /= summ
    return dict


def draw_wordclouds(df):
    prior_dicts, prior_ps = pickle.load(open('prior_dicts.pkl', 'rb'))

    for si, sent in enumerate(['negative', 'neutral', 'positive']):
        commons = commons_dict(df.loc[df['sentiment'] == si].text)

        posterior = [
            (key, commons[key] / (prior_dicts[si][key] if key in prior_dicts[si] else prior_ps[si]))
            for key in commons]
        best_posterior = sorted(posterior, key=lambda x: -x[1])[:40]

        minimal_p = min([p for word, p in best_posterior])
        best_posterior = [(word, int(round(p / minimal_p))) for word, p in best_posterior]
        cloudstrings = [el for (word, n) in best_posterior for el in [word] * n]
        random.shuffle(cloudstrings)

        print(sent)

        wordcloud = WordCloud(max_words=len(cloudstrings)).generate(' '.join(cloudstrings))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def execute(tag):
    df = predict_for_hashtag(tag)
    #     df.date = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['hour'] = df.date.apply(lambda x: x.replace(minute=0, second=0))
    df['sentiment_desc'] = df.sentiment.apply(sentiment_to_desc)
    plot_num(df, tag)
    draw_wordclouds(df)


def run():
    button = widgets.Button(description='Process..')
    text = widgets.Text(value='', description='Tag:', )

    out = widgets.Output()

    def on_button_clicked(_):
        with out:
            clear_output()
            if text.value != '':
                execute(text.value)

    button.on_click(on_button_clicked)
    return widgets.VBox([widgets.HBox([text, button]), out])

