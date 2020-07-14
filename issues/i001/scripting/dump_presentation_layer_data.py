import re
import pandas
import pandas as pd
import argparse
import ast
import random
import jsonplus as json
from pytz import timezone
import pytz
import datetime

parser = argparse.ArgumentParser(description="Run Dump Data Script")
parser.add_argument(
    "--infile", type=str, help="the absolute path to the input data file to run on"
)
parser.add_argument(
    "--chart_id",
    type=str,
    help="the chart_id corresponding to the chart for which you want to produce PL data for",
)


emotions = [
    "neutral",
    "sadness",
    "boredom",
    "anger",
    "empty",
    "enthusiasm",
    "fun",
    "happiness",
    "hate",
    "love",
    "relief",
    "surprise",
    "worry",
]


def _myconverter(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()


def parse_total(t):
    try:
        return pd.Series(ast.literal_eval(t))
    except:
        return pd.Series([None] * len(emotions))


def contains_search_term(series):
    return series.str.contains(
        "coronavirus|covid|corona|sars-cov-2", flags=re.IGNORECASE, regex=True
    )


def convert_score_from_percentage_to_decimal(row):
    for emotion in emotions:
        row[emotion] = row[emotion] / 100
    return row


def apply_rescaling_to_100(emotions_values):
    factor = 100 / emotions_values.sum()
    for e, ev in zip(emotions_values.index.values, emotions_values.values):
        emotions_values[e] = ev * factor
    return emotions_values


def get_emotions_from_total(df):
    return df['total'].apply(parse_total)


emotion_types_to_emotions = {
    "negative": ["sadness", "anger", "empty", "hate", "worry"],
    "neutral": ["neutral", "boredom", "surprise"],
    "positive": ["enthusiasm", "fun", "happiness", "love", "relief"],
}

def get_factor(row, columns):
    try:
        return 100 / row[columns].sum()
    except ZeroDivisionError:
        return 1


class DumpPresentationLayer:
    def __init__(self, infile):
        self.infile = infile
        self.df = pd.read_csv(infile, parse_dates=["date"]).dropna(subset=['text'])
        self.df = self.df[~self.df["total"].isna()]
        self.df_covid_related_terms = self.df[contains_search_term(self.df["text"])]
        self.df[emotions] = get_emotions_from_total(self.df)
        self.df_covid_related_terms[emotions] = get_emotions_from_total(self.df_covid_related_terms)
    
    def get_emotion_types_values(self):
        for emotion_type, e in emotion_types_to_emotions.items():
            self.df[emotion_type] = self.df[e].sum(axis=1)
            self.df_covid_related_terms[emotion_type] = self.df_covid_related_terms[e].sum(axis=1)
        
        #self.df['factor'] = self.df.apply(get_factor, columns=list(emotion_types_to_emotions.keys()), axis=1)
        #self.df_covid_related_terms['factor'] = self.df_covid_related_terms.apply(get_factor, columns=list(emotion_types_to_emotions.keys()), axis=1)
        
        #import pdb; pdb.set_trace()
        #for emotion_type in emotion_types_to_emotions.keys():
        #    self.df[e] = self.df['factor'] * self.df[e]
        #    self.df_covid_related_terms[e] = self.df_covid_related_terms['factor'] * self.df_covid_related_terms[e]

    def get_emotions_values(self):
        #self.df['factor'] = self.df.apply(get_factor, columns=emotions, axis=1)
        #self.df_covid_related_terms['factor'] = self.df_covid_related_terms.apply(get_factor, columns=emotions, axis=1)
        #
        #for e in emotions:
        #    self.df[e] = self.df['factor'] * self.df[e]
        #    self.df_covid_related_terms[e] = self.df_covid_related_terms['factor'] * self.df_covid_related_terms[e]
        pass

    def get_mean_emotions_values(self):
        self.emotions_values = apply_rescaling_to_100(self.df[emotions].mean())
        self.emotions_values_covid_related_terms = apply_rescaling_to_100(self.df_covid_related_terms[emotions].mean())

    def get_mean_emotion_types_values(self):
        self.emotion_types_values = pd.Series()
        self.emotion_types_values_covid_related_terms = pd.Series()
        for emotion_type, emotions in emotion_types_to_emotions.items():
            self.emotion_types_values = self.emotion_types_values.append(
                pd.Series([self.emotions_values[emotions].sum()], index=[emotion_type])
            )
            self.emotion_types_values_covid_related_terms = self.emotion_types_values_covid_related_terms.append(
                pd.Series(
                    [self.emotions_values_covid_related_terms[emotions].sum()],
                    index=[emotion_type],
                )
            )
        
        self.emotion_types_values = self.emotion_types_values.sort_values(ascending=False)
        self.emotion_types_values_covid_related_terms = self.emotion_types_values_covid_related_terms.sort_values(
            ascending=False)
    
    def get_average_tweet_for_group(self, values, df, emotion):
        values_range = (values[emotion] * 0.9, values[emotion] * 1.1)
    
        df = df[(df[emotion] >= values_range[0]) & (df[emotion] <= values_range[1])]

        return df[df['text'].apply(lambda t: len(t) < 100)].sample(1, random_state=1).text.values[0]


    def normalise_datetimes(self):
        self.df['date'] = self.df['date'].apply(lambda dt_obj: pytz.utc.localize(dt_obj))

        def convert_timezone(row):

            location_to_timezone = {
                'NYC-GEO': 'US/Eastern',
                'London-GEO': 'Europe/London',
            }
            row['date'] = row['date'].astimezone(timezone(location_to_timezone[row['location']]))
    
            return row

        # change nyc times to easterm time
        self.df = self.df.apply(convert_timezone, axis=1)
        self.df['date'] = self.df['date'].apply(lambda d: d.date())

    def groupby_date(self):
        self.emotions_values_by_date = self.df.groupby(['date']).agg({
            'neutral': 'mean',
            'sadness': 'mean',
            'boredom': 'mean',
            'anger': 'mean',
            'empty': 'mean',
            'enthusiasm': 'mean',
            'fun': 'mean',
            'happiness': 'mean',
            'hate': 'mean',
            'love': 'mean',
            'relief': 'mean',
            'surprise': 'mean',
            'worry': 'mean',
         }).reset_index()
        
    def make_column_rolling(self, column):
        self.emotions_values_by_date[column] = self.emotions_values_by_date[column].rolling(14).mean()
        self.emotions_values_by_date = self.emotions_values_by_date[~self.emotions_values_by_date[column].isna()]


def dump_for_hb_a(infile):
    dpl = DumpPresentationLayer(infile)
    dpl.get_mean_emotions_values()
    dpl.get_emotion_types_values()
    dpl.get_mean_emotion_types_values()

    average_tweet = dpl.get_average_tweet_for_group(dpl.emotions_values, dpl.df, 'worry')
    average_covid_tweet = dpl.get_average_tweet_for_group(dpl.emotions_values, dpl.df_covid_related_terms, 'worry')
    data = pd.concat([dpl.emotion_types_values_covid_related_terms, dpl.emotion_types_values], axis=1).to_dict(orient='index')
    data_series = []
    for key, value in data.items():
        data_series.append({'name': key, 'x': [v / 100 for v in value.values()]})
    with open('../data/pl-hb-a.json', 'w') as outfile:
        data = json.dumps({
            "y": [
                "An Average COVID-19 Tweet",
                "An Average Tweet",
            ],
            "commments": [
                f'"{average_covid_tweet}"',
                f'"{average_tweet}"'
            ],
            "data_series": data_series
        })
        outfile.write(data)


def dump_for_hb_b(infile):
    dpl = DumpPresentationLayer(infile)
    dpl.get_mean_emotions_values()

    dpl2 = DumpPresentationLayer(infile)
    dpl2.get_mean_emotions_values()
    dpl2.get_emotion_types_values()
    dpl2.get_mean_emotion_types_values()

    average_tweet = dpl.get_average_tweet_for_group(dpl.emotions_values, dpl.df, 'worry')
    average_covid_tweet = dpl.get_average_tweet_for_group(dpl.emotions_values, dpl.df_covid_related_terms, 'worry')

    data = dpl.emotions_values[["worry", "sadness", "hate", "empty", "anger"]].append(dpl2.emotion_types_values[["neutral", "positive"]])
    data_covid = dpl.emotions_values_covid_related_terms[["worry", "sadness", "hate", "empty", "anger"]].append(dpl2.emotion_types_values_covid_related_terms[["neutral", "positive"]])

    data = pd.concat([data_covid, data], axis=1).to_dict(orient='index')
    data_series = []
    for key, value in data.items():
        data_series.append({'name': key, 'x': [v / 100 for v in value.values()]})
    with open('../data/pl-hb-b.json', 'w') as outfile:
        data = json.dumps({
            "y": [
                "An Average COVID-19 Tweet",
                "An Average Tweet",
            ],
            "commments": [
                f'"{average_covid_tweet}"',
                f'"{average_tweet}"'
            ],
            "data_series": data_series
        })
        outfile.write(data)


def dump_for_ts_a(infile):
    dpl = DumpPresentationLayer(infile)
    dpl.normalise_datetimes()
    dpl.groupby_date()
    dpl.make_column_rolling(column='worry')
    with open('../data/pl-ts-a.json', 'w') as outfile:
        data = json.dumps({
            'dates': dpl.emotions_values_by_date['date'].tolist(),
            'data_series': [
                {"name": "worry", "y": dpl.emotions_values_by_date['worry'].apply(lambda v: v / 100).tolist()}
                ]
        })
        outfile.write(data)


def dump_for_ts_b(infile):
    dpl = DumpPresentationLayer(infile)
    dpl.normalise_datetimes()
    dpl.groupby_date()
    dpl.make_column_rolling(column='worry')
    with open('../data/pl-ts-a.json', 'w') as outfile:
        data = json.dumps({
            'dates': dpl.emotions_values_by_date['date'].tolist(),
            'data_series': [
                {"name": "worry", "y": dpl.emotions_values_by_date['worry'].apply(lambda v: v / 100).tolist()}
                ],
        })
        outfile.write(data)


chart_id_fn_map = {
    "hb-a": dump_for_hb_a,
    "hb-b": dump_for_hb_b,
    "ts-a": dump_for_ts_a,
    "ts-b": dump_for_ts_b,
}

if __name__ == "__main__":
    args = parser.parse_args()
    infile = args.infile
    chart_id = args.chart_id
    chart_id_fn_map[chart_id](infile)
