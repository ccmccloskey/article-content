import argparse
import re
import pandas as pd

# change this to form sentiment import SentimentAnalysis
from twitter.sentiment.sentiment import SentimentAnalysis


parser = argparse.ArgumentParser(description="Run Dump Data Script")
parser.add_argument(
    "--infile", type=str, help="the absolute path to the input data file to run on"
)
parser.add_argument(
    "--outfile", type=str, help="the absolute path to the file to output the data to"
)
parser.add_argument(
    "--sample",
    type=float,
    help="proportion of data to sample for algorithm (random_state is persisted)",
    default=0.05,
)


def get_total_score(text):
    try:
        SA = SentimentAnalysis(text)
        SA.analyze()
        return SA.get_total_score()
    except:
        return None
    return


def preprocess_text(text):
    return " ".join(
        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()
    )


if __name__ == "__main__":
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    sample = args.sample

    df = pd.read_csv(infile, parse_dates=["date"])
    df = df.sample(frac=sample, replace=False, random_state=1).reset_index()
    df = df[~df["text"].isna()]
    # preprocess text before running nlp
    df["text"] = df["text"].apply(preprocess_text)
    # run nlp sentiment analyzer from sentiment-anlysis package
    with open(outfile, "w", newline="") as f:
        df["total"] = df["text"].apply(get_total_score)
        df.to_csv(f)
