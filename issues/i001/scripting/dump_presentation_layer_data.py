import re
import pandas
import pandas as pd
import argparse
import ast

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


def get_data_hb_a():
    df = pd.read_csv(infile, parse_dates=["date"]).drop(columns="Unnamed: 0")
    df = df[~df["total"].isna()]
    df_covid_related_terms = df[contains_search_term(df["text"])]

    # get emotions from total
    df[emotions] = df["total"].apply(parse_total)
    df_covid_related_terms[emotions] = df_covid_related_terms["total"].apply(
        parse_total
    )

    # 1 - Emotion Types
    # Algo produces a score per tweet for each emotion. The % liklihood to which a tweet expresses a given emotion is output.
    emotions_values = df[emotions].mean()
    emotions_values_covid_related_terms = df_covid_related_terms[emotions].mean()

    # rescale factor due to rounding errors to get total out of 100
    emotions_values = apply_rescaling_to_100(emotions_values)
    emotions_values_covid_related_terms = apply_rescaling_to_100(
        emotions_values_covid_related_terms
    )

    emotions_values = convert_score_from_percentage_to_decimal(emotions_values)
    emotions_values_covid_related_terms = convert_score_from_percentage_to_decimal(
        emotions_values_covid_related_terms
    )

    emotion_types_values = pd.Series()
    emotion_types_values_covid_related_terms = pd.Series()
    for emotion_type, emotions in emotion_types_to_emotions.items():
        emotion_types_values = emotion_types_values.append(
            pd.Series([emotions_values[emotions].sum()], index=[emotion_type])
        )
        emotion_types_values_covid_related_terms = emotion_types_values_covid_related_terms.append(
            pd.Series(
                [emotions_values_covid_related_terms[emotions].sum()],
                index=[emotion_type],
            )
        )


chart_id_fn_map = {
    "hb-a": get_data_hb_a,
}


if __name__ == "__main__":
    args = parser.parse_args()
    infile = args.infile
    chart_id = args.chart_id
    chart_id_fn_map[chart_id]()
