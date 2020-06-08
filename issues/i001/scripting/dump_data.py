import csv
import datetime as dt
import time

import GetOldTweets3 as got
import argparse

parser = argparse.ArgumentParser(description="Run Dump Data Script")
parser.add_argument(
    "--search", type=str, help="the search query given for twitter /search"
)
parser.add_argument(
    "--outfile", type=str, help="the absolute path to the file to output the data to"
)

# look at last 2 months
start_date = dt.datetime(2020, 5, 31, 0, 0, 0)
end_date = dt.datetime(2020, 5, 31, 0, 0, 0)


class Location:
    def __init__(self, name, coordinates, within):
        self.name = name
        self.coordinates = coordinates
        self.within = within


# compare capital cities
london = Location("London-GEO", "51.53, -0.38", "20km")
nyc = Location("NYC-GEO", "40.70, -74.26", "20km")


def get_all_data_for_query_search(start_date, end_date, search, locations, out):

    with open(out, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator="\n")
        csv_writer.writerow(
            [
                "date",
                "username",
                "to",
                "replies",
                "retweets",
                "favorites",
                "text",
                "geo",
                "mentions",
                "hashtags",
                "id",
                "permalink",
                "location",
            ]
        )

        initial_criteria = got.manager.TweetCriteria().setQuerySearch(search)

        initial_time = time.time()

        for location in locations:

            criteria_with_location = initial_criteria.setNear(
                location.coordinates
            ).setWithin(location.within)
            start_date_for_location = start_date

            while start_date_for_location <= end_date:
                since, until = (
                    start_date_for_location.strftime("%Y-%m-%d"),
                    (start_date_for_location + dt.timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    ),
                )
                tweet_criteria = (
                    criteria_with_location.setSince(since)
                    .setUntil(until)
                    .setMaxTweets(max_tweets)
                )
                tweets = got.manager.TweetManager.getTweets(tweet_criteria)

                print(
                    f'got {len(tweets)} tweets with search "{search}" for {location.name} between {since} and {until}'
                )

                for t in tweets:
                    data = [
                        t.date.strftime("%Y-%m-%d %H:%M:%S"),
                        t.username,
                        t.to or "",
                        t.replies,
                        t.retweets,
                        t.favorites,
                        t.text,
                        t.geo,
                        t.mentions,
                        t.hashtags,
                        t.id,
                        t.permalink,
                        location.name,
                    ]

                    csv_writer.writerow(map(str, data))

                start_date_for_location += dt.timedelta(days=1)

                time_now = time.time() - initial_time

                print(f"{time_now} seconds have elapsed")


if __name__ == "__main__":
    args = parser.parse_args()
    search = args.search
    out = args.outfile
    get_all_data_for_query_search(start_date, end_date, search, [london, nyc], out)
