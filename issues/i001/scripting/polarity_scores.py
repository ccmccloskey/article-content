# from pandarallel import pandarallel
import ast
import calendar
# import nltk
import csv
# from textblob import TextBlob
# from googletrans import Translator
import datetime as dt
import math
import os
import pdb
import random
import re
import statistics
import time
from collections import Counter
from enum import Enum

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
from dateutil.relativedelta import relativedelta
from pytz import timezone

## PLOTLY CONFIG


# nltk.download('punkt')
# pandarallel.initialize()

# from twitter.sentiment.sentiment import SentimentAnalysis, NotSupportLanguageError

os.chdir("/Users/ciaranmccloskey/Documents/projects/open-source/issues/i001/scripting")
# df = pd.read_csv('no-search.csv', parse_dates=['date'],
# dtype={
#    'username': np.dtype('object'),
#    'to': np.dtype('object'),
#    'replies': np.dtype('int64'),
#    'retweets': np.dtype('int64'),
#    'favorites': np.dtype('int64'),
#    'text': np.dtype('object'),
#    'geo': np.dtype('float64'),
#    'mentions': np.dtype('object'),
#    'hashtags': np.dtype('object'),
#    'id': np.dtype('int64'),
#    'permalink': np.dtype('object'),
#    'location': np.dtype('object')
#    }
# )

# sent = SentimentAnalysis('this is my way of saying I understand the model').analyze()


class Emotions:
    neutral = "neutral"
    sadness = "sadness"
    boredom = "boredom"
    anger = "anger"
    empty = "empty"
    enthusiasm = "enthusiasm"
    fun = "fun"
    happiness = "happiness"
    hate = "hate"
    love = "love"
    relief = "relief"
    surprise = "surprise"
    worry = "worry"


emotions = [
    Emotions.neutral,
    Emotions.sadness,
    Emotions.boredom,
    Emotions.anger,
    Emotions.empty,
    Emotions.enthusiasm,
    Emotions.fun,
    Emotions.happiness,
    Emotions.hate,
    Emotions.love,
    Emotions.relief,
    Emotions.surprise,
    Emotions.worry,
]


class EmotionTypes:
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


emotion_types = [
    EmotionTypes.positive,
    EmotionTypes.negative,
    EmotionTypes.neutral,
]


class Columns:
    rescaler = "rescaler"
    date = "date"
    text = "text"
    count = "count"
    total = "total"


class SentimentColorPalette(Enum):
    red = "rgb(103,0,13)"
    black = "rgb(0,0,0)"
    green = "rgb(0,68,27)"

    @classmethod
    def to_list(cls):
        return [color.value for color in cls]


class DefaultSeriesColorPalette(Enum):
    first = "rgb(0,63,92)"
    second = "rgb(88,80,141)"
    third = "rgb(188,80,144)"
    fourth = "rgb(255,99,97)"
    fifth = "rgb(255,166,0)"

    @classmethod
    def to_list(cls):
        return [color.value for color in cls]


class NeutralSeriesColorPalette(Enum):
    white = "white"
    grey = "rgb(217,217,217)"

    @classmethod
    def to_value_list(cls):
        return [color.value for color in cls]


def preprocess_text(text):
    return " ".join(
        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()
    )


def get_polarity_score(text):
    return TextBlob(text).sentiment.polarity


def get_total_score(text):
    try:
        SA = SentimentAnalysis(text)
        SA.analyze()
        return SA.get_total_score()
    except:
        return None
    return


def calculate_percent_delta(initial_value, current_value):
    return current_value - initial_value


def parse_total(t):
    try:
        return pd.Series(ast.literal_eval(t))
    except:
        return pd.Series([None] * len(emotions))


## take a sample from data to save time in algo stage
# df = df.sample(frac=0.05, replace=False, random_state=1).reset_index()
# print(Counter(df['date'].apply(lambda dt_obj: dt_obj.date()).tolist()))
# with open(f'no-search-sample.csv', 'w', newline='') as f:
#    df.to_csv(f)
#
# df = pd.read_csv('no-search-sample.csv', parse_dates=['date'])
# df = df[~df['text'].isna()]
## preprocess text before running nlp
# df['text'] = df['text'].apply(preprocess_text)
#
# istart, iend = 15000, 20000
#
# df = df.iloc[istart:iend]
### run nlp sentiment analyzer from sentiment-anlysis package
# with open(f'data-no-search-sample-{istart}-{iend}.csv', 'w', newline='') as f:
#    df['total'] = df['text'].apply(get_total_score)
#    df.to_csv(f)

################################### Post Algorithm ###################################

# categorise emotions into positive, negative, netural and sum the values in these
emotion_types_to_emotions = {
    "negative": ["sadness", "anger", "empty", "hate", "worry"],
    "neutral": ["neutral", "boredom", "surprise"],
    "positive": ["enthusiasm", "fun", "happiness", "love", "relief"],
}


def contains_search_term(series):
    return series.str.contains(
        "coronavirus|covid|corona|sars-cov-2", flags=re.IGNORECASE, regex=True
    )


## pre content layer
df = pd.read_csv("../data/no-search-sample-with-scores.csv", parse_dates=["date"]).drop(
    columns="Unnamed: 0"
)
df = df[~df["total"].isna()]
df_covid_related_terms = df[contains_search_term(df["text"])]

# get emotions from total
df[emotions] = df["total"].apply(parse_total)
df_covid_related_terms[emotions] = df_covid_related_terms["total"].apply(parse_total)


# 1 - Emotion Types
# Algo produces a score per tweet for each emotion. The % liklihood to which a tweet expresses a given emotion is output.
emotions_values = df[emotions].mean()
emotions_values_covid_related_terms = df_covid_related_terms[emotions].mean()

# a) breakdown of emotions across entire data


def convert_score_from_percentage_to_decimal(row):
    for emotion in emotions:
        row[emotion] = row[emotion] / 100
    return row


def apply_rescaling_to_100(emotions_values):
    factor = 100 / emotions_values.sum()
    for e, ev in zip(emotions_values.index.values, emotions_values.values):
        emotions_values[e] = ev * factor
    return emotions_values


# rescale factor due to rounding errors to get total out of 100
emotions_values = apply_rescaling_to_100(emotions_values)
emotions_values_covid_related_terms = apply_rescaling_to_100(
    emotions_values_covid_related_terms
)

emotions_values = convert_score_from_percentage_to_decimal(emotions_values)
emotions_values_covid_related_terms = convert_score_from_percentage_to_decimal(
    emotions_values_covid_related_terms
)


# emotion_types_data
emotion_types_values = pd.Series()
emotion_types_values_covid_related_terms = pd.Series()
for emotion_type, emotions in emotion_types_to_emotions.items():
    emotion_types_values = emotion_types_values.append(
        pd.Series([emotions_values[emotions].sum()], index=[emotion_type])
    )
    emotion_types_values_covid_related_terms = emotion_types_values_covid_related_terms.append(
        pd.Series(
            [emotions_values_covid_related_terms[emotions].sum()], index=[emotion_type]
        )
    )

# sort by largest value
emotion_types_values = emotion_types_values.sort_values(ascending=False)
emotion_types_values_covid_related_terms = emotion_types_values_covid_related_terms.sort_values(
    ascending=False
)


# color map
emotion_types_to_colors = {
    "negative": "rgb(239,59,44)",
    "neutral": "rgba(115,115,115,0.1)",
    "positive": "rgba(65,171,93,0.1)",
}
#


def create_horizontal_bar_emotion_types():
    fig = go.Figure()
    # FIG a) - horizontal bar - emotion-types
    fig = go.Figure(
        data=[
            go.Bar(
                name=et_name,
                y=["An Average Tweet", "An Average COVID-19 Tweet"],
                x=[et, et_st],
                width=[0.2, 0.2],
                marker_color=emotion_types_to_colors[et_name],
                orientation="h",
            )
            for et_name, et, et_st in zip(
                emotion_types_values.index.values,
                emotion_types_values.values,
                emotion_types_values_covid_related_terms.values,
            )
        ]
    )

    n_series = len([emotion_types_values, emotion_types_values_covid_related_terms])

    annotations = []

    def boldify_text(text):
        return "<b>" + text + "</b>"

    def italicize_text(text):
        return "<i>" + text + "</i>"

    # series titles
    yref_uplift = 1 / n_series
    yref_downlift = (1 / n_series) * 0.25
    yref_value = 0
    texts = ["An Average Tweet", "An Average COVID-19 Tweet"]
    capitalized_texts = [t.upper() for t in texts]
    boldified_texts = [boldify_text(t) for t in capitalized_texts]

    # tweet annotations
    tweet_texts = [
        '"Trump has not got a clue, we have a true clown as leader, I am worried."',
        '"They have closed the park near to me. Our civil liberties are being slowly ripped from under us. We should be scared. Thanks COVID."',
    ]
    italicized_tweet_texts = [italicize_text(t) for t in tweet_texts]

    for text, tweet_text in zip(boldified_texts, italicized_tweet_texts):
        yref_value += yref_uplift
        annotations.append(
            dict(
                xref="x",
                x=0.5,
                yref="paper",
                y=yref_value,
                xanchor="center",
                yanchor="middle",
                text=text,
                font=dict(family="Roboto", size=16, color="rgb(8,48,107)"),
                showarrow=False,
            )
        )

        annotations.append(
            dict(
                xref="x",
                x=0.5,
                yref="paper",
                y=yref_value - yref_downlift,
                xanchor="center",
                yanchor="middle",
                text=tweet_text,
                font=dict(family="Roboto Thin", size=20, color="rgb(8,48,107)"),
                showarrow=False,
            )
        )

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            tickformat=",.0%",
            # hoverformat='%b %d',
            tickvals=[0, 0.25, 0.50, 0.75, 1],
        ),
        yaxis=dict(
            showgrid=False,
            # zeroline=False,
            # showline=True,
            showticklabels=False,
            # tickfont=dict(family='Roboto', size=16, color='rgb(217,217,217)'),
            # tickvals=np.arange(0, 1, 0.01),
            # hoverformat=',.1%',
        ),
        # autosize=False,
        # margin=dict(
        #    autoexpand=False,
        #    l=100,
        #    r=20,
        #    t=110,
        # ),
        xaxis_title="Sentiment Breakdown",
        showlegend=False,
        plot_bgcolor="white",
        barmode="stack",
        font=dict(family="Roboto", size=16, color="rgb(8,48,107)"),
        annotations=annotations,
    )
    return fig


# def create_horizontal_bar_emotions(emotions_values, emotions_values_covid_related_terms):
#
#    def create_data_presentation_layer_data(values_data, types_values_data):
#
#        emotions_negative = values_data[emotion_types_to_emotions['negative']].sort_values(ascending=False)
#        emotion_types_neutral_positive = types_values_data[['neutral', 'positive']]
#        emotions_negative_with_emotion_types_neutral_positive = emotions_negative.append(emotion_types_neutral_positive)
#        return emotions_negative, emotions_negative_with_emotion_types_neutral_positive
#
#    emotions_negative, emotions_negative_with_emotion_types_neutral_positive = create_data_presentation_layer_data(
#        emotions_values, emotion_types_values,
#    )
#
#    emotions_negative_covid_related_terms, emotions_negative_with_emotion_types_neutral_positive_covid_related_terms = create_data_presentation_layer_data(
#        emotions_values_covid_related_terms, emotion_types_values_covid_related_terms,
#    )
#
#    #colors = px.colors.sequential.Reds_r[:5] + [emotion_types_to_colors['neutral'], emotion_types_to_colors['positive']]
#    colors = [
#        'rgba(103,0,13,1)',
#        'rgba(103,0,13,0.9)',
#        'rgba(103,0,13,0.8)',
#        'rgba(103,0,13,0.7)',
#        'rgba(103,0,13,0.6)'] + [emotion_types_to_colors['neutral'], emotion_types_to_colors['positive']]
#    #import pdb; pdb.set_trace()
#
#    fig = go.Figure(data=[
#        go.Bar(
#            name=et_name,
#            y=['An Average Tweet', 'An Average COVID-19 Tweet'],
#            x=[et, et_st],
#            width=[0.2, 0.2],
#            marker_color=c,
#            orientation='h') for et_name, et, et_st, c in zip(
#                emotions_negative_with_emotion_types_neutral_positive.index.values,
#                emotions_negative_with_emotion_types_neutral_positive.values,
#                emotions_negative_with_emotion_types_neutral_positive_covid_related_terms.values,
#                colors,
#                )
#    ])
#
#    n_series = len([emotion_types_values, emotion_types_values_covid_related_terms])
#
#    annotations = []
#
#    def boldify_text(text):
#        return '<b>' + text + '</b>'
#
#
#    def italicize_text(text):
#        return '<i>' + text + '</i>'
#
#    # series titles
#    yref_uplift = 1 / n_series
#    yref_downlift = (1 / n_series) * 0.25
#    yref_value = 0
#    texts = ['An Average Tweet', 'An Average COVID-19 Tweet']
#    capitalized_texts = [t.upper() for t in texts]
#    boldified_texts = [boldify_text(t) for t in capitalized_texts]
#
#    # tweet annotations
#    tweet_texts = [
#        '"Trump has not got a clue, we have a true clown as leader, I am worried."',
#        '"They have closed the park near to me. Our civil liberties are being slowly ripped from under us. We should be scared. Thanks COVID."'
#    ]
#    italicized_tweet_texts = [italicize_text(t) for t in tweet_texts]
#
#    for text, tweet_text in zip(boldified_texts, italicized_tweet_texts):
#        yref_value += yref_uplift
#        annotations.append(dict(
#            xref='x', x=0.5, yref='paper', y=yref_value,
#            xanchor='center', yanchor='middle', text=text,
#            font=dict(family='Roboto', size=16, color='rgb(8,48,107)'),
#            showarrow=False))
#
#        annotations.append(dict(
#            xref='x', x=0.5, yref='paper', y=yref_value - yref_downlift,
#            xanchor='center', yanchor='middle', text=tweet_text,
#            font=dict(family='Roboto Thin', size=20, color='rgb(8,48,107)'),
#            showarrow=False))
#
#
#
#    fig.update_layout(
#            xaxis=dict(
#                showline=True,
#                showgrid=False,
#                showticklabels=True,
#                zeroline=False,
#                tickformat=',.0%',
#                #hoverformat='%b %d',
#                tickvals=[0, 0.25, 0.50, 0.75, 1],
#            ),
#            yaxis=dict(
#                showgrid=False,
#                #zeroline=False,
#                #showline=True,
#                showticklabels=False,
#                #tickfont=dict(family='Roboto', size=16, color='rgb(217,217,217)'),
#                #tickvals=np.arange(0, 1, 0.01),
#                #hoverformat=',.1%',
#            ),
#            #autosize=False,
#            #margin=dict(
#            #    autoexpand=False,
#            #    l=100,
#            #    r=20,
#            #    t=110,
#            #),
#            xaxis_title="Emotion Breakdown",
#            showlegend=False,
#            plot_bgcolor='white',
#            barmode='stack',
#            font=dict(family='Roboto', size=16, color='rgb(8,48,107)'),
#            annotations=annotations
#            )
#    return fig
#
#
fig = create_horizontal_bar_emotion_types()
fig.show()


pdb.set_trace()
#
##fig = create_horizontal_bar_emotions(emotions_values, emotions_values_covid_related_terms)
##fig.show()


# fig.write_image(f"images/nosearch/emotion-types.png")
## large proportion of negative emotion - make comparison of this to running on a basic tweet. Look at negative in more granular breakdown
#
## b) breakdown of emotions across entire data - with furhter
# emotions_negative = emotions_values[emotion_types_to_emotions['negative']].sort_values(ascending=False)
# emotion_types_neutral_positive = emotion_types_values[['neutral', 'positive']]
# emotions_negative_with_emotion_types_neutral_positive = emotions_negative.append(emotion_types_neutral_positive)
#
## colors
# colors = px.colors.sequential.Reds_r[:len(emotions_negative)] + [emotion_types_to_colors['neutral'], emotion_types_to_colors['positive']]

# FIG b) - horizontal bar - emotion-types-neutral_positive and emotions-negative
# fig = go.Figure(data=[
#    go.Bar(
#        name=et_name,
#        y=[''],
#        x=[et],
#        marker_color=c,
#        orientation='h') for et_name, et, c in zip(
#            emotions_negative_with_emotion_types_neutral_positive.index.values,
#            emotions_negative_with_emotion_types_neutral_positive.values,
#            colors
#            )
# ])
# fig.update_layout(barmode='stack')
# fig.write_image(f"images/nosearch/emotion-types-neutral-positive-emotions-negative.png")


# 2 - Emotion Types Time Series
# Look at whether there is fluctutation over time.
# a) Emotion Types All Data

# calculate emotion types values per tweet
# for emotion_type, emotions in emotion_types_to_emotions.items():
#    df[emotion_type] = df[emotions].sum(axis=1)
#
## groupby date
##df['date'] = df['date'].apply(lambda d: d.date())
##emotion_types_values_by_date = df.groupby('date').agg({
##    'negative': 'mean',
##    'neutral': 'mean',
##    'positive': 'mean',
##}).reset_index()
#
# df['date'] = df['date'].apply(lambda dt_obj: pytz.utc.localize(dt_obj))
#
# def convert_timezone(row):
#
#    location_to_timezone = {
#        'NYC-GEO': 'US/Eastern',
#        'London-GEO': 'Europe/London',
#        'Seoul-GEO': 'Asia/Seoul',
#    }
#    row['date'] = row['date'].astimezone(timezone(location_to_timezone[row['location']]))
#
#    return row
#
## change nyc times to easterm time
# df = df.apply(convert_timezone, axis=1)
#
## create hour of the day featuere
##df['hour'] = df['date'].apply(lambda dt_obj: dt_obj.hour)
#
## select date part
# df['date'] = df['date'].apply(lambda d: d.date())
#
## groupby hour of the day
##emotions_values_by_hour_of_day = df.groupby('date').agg({
##    'neutral': 'mean',
##    'sadness': 'mean',
##    'boredom': 'mean',
##    'anger': 'mean',
##    'empty': 'mean',
##    'enthusiasm': 'mean',
##    'fun': 'mean',
##    'happiness': 'mean',
##    'hate': 'mean',
##    'love': 'mean',
##    'relief': 'mean',
##    'surprise': 'mean',
##    'worry': 'mean',
##}).reset_index()
#
# def apply_rescaling_to_100(row):
#    factor = 100 / row.sum()
#    for emotion_type in emotion_types:
#        row[emotion_type] = factor * row[emotion_type]
#    return row
#
#
##
### apply rescaler
##emotion_types_values_by_date[emotion_types] = emotion_types_values_by_date[emotion_types].apply(apply_rescaling_to_100, axis=1)
##
### FIG a) - area - emotion types time series
##fig = go.Figure()
##
##for emotion_type, color in emotion_types_to_colors.items():
##    fig.add_trace(go.Scatter(
##        x=emotion_types_values_by_date['date'],
##        y=emotion_types_values_by_date[emotion_type],
##        hoverinfo='x+y',
##        mode='lines',
##        line=dict(width=0.5, color=color),
##        stackgroup='one' # define stack group
##    ))
##
##fig.update_layout(yaxis_range=(0, 100))
##fig.write_image(f"images/nosearch/emotion-types-timeseries.png")
###
###
### apply 7 day rolling mean to smooth noise
##emotion_types_values_by_date[emotion_types] = emotion_types_values_by_date[emotion_types].rolling(7).mean()
##emotion_types_values_by_date = emotion_types_values_by_date.dropna()
##
### FIG b) - area - emotion types time series 7 day rolling mean
##fig = go.Figure()
##
##for emotion_type, color in emotion_types_to_colors.items():
##    fig.add_trace(go.Scatter(
##        x=emotion_types_values_by_date['date'],
##        y=emotion_types_values_by_date[emotion_type],
##        hoverinfo='x+y',
##        mode='lines',
##        line=dict(width=0.5, color=color),
##        stackgroup='one' # define stack group
##    ))
##
##fig.update_layout(yaxis_range=(0, 100))
##fig.write_image(f"images/nosearch/emotion-types-timeseries-7day-rolling.png")
##
### c) emotions area time series
#
# emotions = [
#    Emotions.neutral,
#    Emotions.sadness,
#    Emotions.boredom,
#    Emotions.anger,
#    Emotions.empty,
#    Emotions.enthusiasm,
#    Emotions.fun,
#    Emotions.happiness,
#    Emotions.hate,
#    Emotions.love,
#    Emotions.relief,
#    Emotions.surprise,
#    Emotions.worry,
# ]
#
#
# df[emotions] = df[emotions].apply(convert_score_from_percentage_to_decimal, axis=1)
#
# emotions_values_by_date = df.groupby(['date']).agg({
#    'neutral': 'mean',
#    'sadness': 'mean',
#    'boredom': 'mean',
#    'anger': 'mean',
#    'empty': 'mean',
#    'enthusiasm': 'mean',
#    'fun': 'mean',
#    'happiness': 'mean',
#    'hate': 'mean',
#    'love': 'mean',
#    'relief': 'mean',
#    'surprise': 'mean',
#    'worry': 'mean',
# }).reset_index()
#
##fig = go.Figure()
##
##for location, fill in zip(['London-GEO', 'NYC-GEO'], ['tozeroy', 'tonexty']):
##    fig.add_trace(go.Scatter(
##        x=emotions_values_by_date[emotions_values_by_date['location'] == location]['date'],
##        y=emotions_values_by_date[emotions_values_by_date['location'] == location]['worry'],
##        hoverinfo='x+y',
##        mode='lines',
##        #line=dict(width=0.5, color='red'),
##        fill=fill # define stack group
##    ))
##
##fig.show()
#
##london_df = emotions_values_by_date[emotions_values_by_date['location'] == 'London-GEO']
##nyc_df = emotions_values_by_date[emotions_values_by_date['location'] == 'NYC-GEO']
#
# fig = go.Figure()
#


class TimeSeries:
    def __init__(self):
        self.xaxis = dict(
            showline=False,
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            tickformat="%b",
            hoverformat="%b %d",
            tickfont=dict(family="Roboto", size=16, color="rgb(217,217,217)"),
        )
        self.yaxis = dict(
            showline=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(family="Roboto", size=16, color="rgb(217,217,217)"),
        )
        self.showlegend = False
        self.plot_bgcolor = "white"

    def update_xaxis_configuration(self, **params):
        for key, value in params.items():
            self.xaxis.update({key: value})

    def update_yaxis_configuration(self, **params):
        for key, value in params.items():
            self.yaxis.update({key: value})

    def update_show_legend(self):
        pass


def create_timeseries_plot():
    k = 14
    annotations = []
    for col, color in zip(["worry"], [SentimentColorPalette.red.value]):

        emotions_values_by_date[col] = emotions_values_by_date[col].rolling(14).mean()
        emotions_values_by_date_complete = emotions_values_by_date[
            ~emotions_values_by_date[col].isna()
        ]

        fig.add_trace(
            go.Scatter(
                x=emotions_values_by_date_complete["date"],
                y=emotions_values_by_date_complete[col],
                mode="lines",
                line_shape="spline",
                name=col.capitalize(),
                line=dict(color="rgba(103,0,13,0.1)", width=4),
            )
        )

        # endpoints
        fig.add_trace(
            go.Scatter(
                x=[emotions_values_by_date_complete["date"].iloc[0]],
                y=[emotions_values_by_date_complete[col].iloc[0]],
                mode="markers",
                marker=dict(color="rgba(103,0,13,0.1)", size=14),
                marker_symbol="square",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[emotions_values_by_date_complete["date"].iloc[-1]],
                y=[emotions_values_by_date_complete[col].iloc[-1]],
                mode="markers",
                marker=dict(color="rgba(103,0,13,0.1)", size=20),
                marker_symbol="hexagon",
                hoverinfo="skip",
            )
        )

        # annotations
        annotations.append(
            dict(
                xref="x",
                x=emotions_values_by_date_complete["date"].iloc[-1],
                yref="y",
                y=emotions_values_by_date_complete[col].iloc[-1],
                xanchor="left",
                xshift=10,
                yanchor="middle",
                text=col.capitalize(),
                font=dict(family="Roboto", size=16, color="rgba(103,0,13,0.1)"),
                showarrow=False,
            )
        )

    unique_year_months = (
        emotions_values_by_date["date"]
        .apply(lambda date: (date.year, date.month))
        .unique()
    )
    colors = NeutralSeriesColorPalette.to_list()

    def get_number_of_days_in_year_month(year, month):
        return calendar.monthrange(year, month)[1]

    def get_middle_datetime_in_year_month(year, month):
        number_of_days_in_year_month = get_number_of_days_in_year_month(year, month)
        first, last = (
            dt.datetime(year, month, 1),
            dt.datetime(year, month, number_of_days_in_year_month),
        )
        return first + ((last - first) / 2)

    # create year month backgrounds
    shapes = []
    middle_datetimes = []
    for i, (year, month) in enumerate(unique_year_months):
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=dt.datetime(year, month, 1).date(),
                y0=0,
                x1=dt.datetime(year, month + 1, 1).date(),
                y1=1,
                fillcolor=colors[i % len(colors)],
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        )
        middle_datetimes.append(get_middle_datetime_in_year_month(year, month))

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            zeroline=True,
            tickformat="%b",
            hoverformat="%b %d",
            tickvals=middle_datetimes,
            tickfont=dict(family="Roboto", size=16, color="rgb(217,217,217)"),
        ),
        yaxis=dict(
            showgrid=False,
            # zeroline=False,
            # showline=True,
            showticklabels=True,
            tickfont=dict(family="Roboto", size=16, color="rgb(217,217,217)"),
            tickvals=np.arange(0, 1, 0.01),
            hoverformat=",.1%",
        ),
        # autosize=False,
        # margin=dict(
        #    autoexpand=False,
        #    l=100,
        #    r=20,
        #    t=110,
        # ),
        showlegend=False,
        plot_bgcolor="white",
        annotations=annotations,
        shapes=shapes,
    )

    if not os.path.exists("images"):
        os.mkdir("images")

    return fig, emotions_values_by_date_complete


class TimeSeriesEvent:
    def __init__(self, date, data, series_col, display_text):
        self.x = date
        self.y = data[data["date"] == date][series_col].iloc[0]
        self.display_text = display_text


def add_events_to_timeseries_plot(fig, events):
    for event in events:
        fig.add_trace(
            go.Scatter(
                x=[event.x],
                y=[event.y],
                mode="markers",
                marker=dict(color=SentimentColorPalette.red.value, size=16),
                marker_symbol="hexagon",
                hoverinfo="skip",
            )
        )
        fig.layout.annotations += (
            dict(
                xref="x",
                x=event.x,
                yref="y",
                y=event.y,
                xanchor="center",
                yanchor="bottom",
                yshift=5,
                text=event.display_text,
                font=dict(
                    family="Roboto", size=14, color=SentimentColorPalette.red.value
                ),
                showarrow=False,
            ),
        )
    return fig


# plot
# fig, emotions_values_by_date_complete = create_timeseries_plot()
#
# fig = add_events_to_timeseries_plot(fig, [
#    TimeSeriesEvent(
#        dt.datetime(2020, 3, 11).date(),
#        emotions_values_by_date_complete,
#        'worry',
#        '<b>WHO declares the outbreak a pandemic</b>'),
#    TimeSeriesEvent(
#        dt.datetime(2020, 3, 31).date(),
#        emotions_values_by_date_complete,
#        'worry',
#        '<b>More than 1/3 of humanity is under some form of lockdown</b>')
#    ])

# fig.show()

# fig.write_image(f"images/nosearch/rolling{k}-{col}-scores-all-tz.png")
