import csv
import datetime as dt
import json
from random import randint

import GetOldTweets3 as got
import pandas as pd

# look at last 2 months
start_date = dt.datetime(2020, 5, 22, 0, 0, 0)
end_date = dt.datetime(2020, 5, 31, 0, 0, 0)


class Location:
    def __init__(self, name, coordinates, within):
        self.name = name
        self.coordinates = coordinates
        self.within = within


# compare capital cities
london = Location("London-GEO", "51.53, -0.38", "20km")
seoul = Location("Seoul-GEO", "37.57, 126.85", "15km")
nyc = Location("NYC-GEO", "40.70, -74.26", "20km")


def get_all_data_for_query_search(start_date, end_date, search, locations):
    import time
    import os

    os.chdir("/Users/ciaranmccloskey/Documents/projects/open-source/001/twitter")

    locations_str = "-".join([l.name for l in locations])

    with open(f"{search}-{locations_str}.csv", "w", newline="") as csvfile:
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
                    .setMaxTweets(2000)
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

                # try to circumvent rate limiting
                # time.sleep(randint(4, 7))


search = ""

if __name__ == "__main__":
    get_all_data_for_query_search(start_date, end_date, search, [nyc])
