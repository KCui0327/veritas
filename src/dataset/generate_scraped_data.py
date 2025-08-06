import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from pygooglenews import GoogleNews


def get_real_news():
    gn = GoogleNews()

    df = pd.DataFrame(columns=["Title", "Date", "Label"])
    topics = ["politics", "nation", "business", "technology", "entertainment", "sports"]
    for topic in topics:
        top = gn.search(topic)
        entries = top["entries"]
        entries_list = [[entry.title, entry.published] for entry in entries]
        df_entry = pd.DataFrame(entries_list, columns=["Title", "Date"])

        df = pd.concat([df, df_entry.iloc[:30, :]], ignore_index=True)
        df.loc[:, "Label"] = "True"

    df["Date"] = df["Date"].apply(
        lambda date: datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %Z").strftime(
            "%B %#d, %Y"
        )
    )
    return df


# The following functions scrape article from prominent fake/satirical news websites
def empire_news_crawler():
    categories = [
        "entertainment/",
        "sports/",
        "politics/",
        "business/",
        "healthfitness/",
        "sciencetech/",
        "world",
    ]
    df = pd.DataFrame(columns=["Title", "Date", "Label"])
    for link in categories:
        response = requests.get("https://empirenews.net/category/{}".format(link))
        soup = bs(response.content, "html5lib")

        # Find a article by its ID
        headers = soup.find_all("header", class_="entry-header")
        header_texts = [header.get_text(strip=True) for header in headers]
        for text in header_texts:
            parts = text.split("Posted on")
            title = parts[0]
            match = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", parts[1])
            date = match.group(1) if match else ""
            df_entry = pd.DataFrame(
                {"Title": [title], "Date": [date], "Label": ["False"]}
            )
            df = pd.concat([df, df_entry], ignore_index=True)

    return df


def now8news_crawler():
    df = pd.DataFrame(columns=["Title", "Label"])
    for page in range(1, 5):
        response = requests.get("https://now8news.com/page/{}/".format(page))
        soup = bs(response.content, "html5lib")

        # Find a article by its class name
        headers = soup.find_all("h3", class_="entry-title content-list-title")
        header_texts = [header.get_text(strip=True) for header in headers]
        for text in header_texts:
            df_entry = pd.DataFrame({"Title": [text], "Label": ["False"]})
            df = pd.concat([df, df_entry], ignore_index=True)

    return df


def the_onion_crawler():
    df = pd.DataFrame(columns=["Title", "Label"])
    for page in range(1, 5):
        response = requests.get("https://theonion.com/news/page/{}/".format(page))
        soup = bs(response.content, "html5lib")

        # Find a article by its class name
        headers = soup.find_all(
            "h3", class_="is-style-scale-2 wp-block-post-title has-delta-font-size"
        )
        header_texts = [header.get_text(strip=True) for header in headers]
        for text in header_texts:
            df_entry = pd.DataFrame({"Title": [text], "Label": ["False"]})
            df = pd.concat([df, df_entry], ignore_index=True)

    return df


# Concat all df into one csv file
df = pd.DataFrame(columns=["Title", "Date", "Label"])

empire_news_df = empire_news_crawler()
now8news_df = now8news_crawler()
the_onion_df = the_onion_crawler()
real_news_df = get_real_news()

df = pd.concat(
    [df, empire_news_df, now8news_df, the_onion_df, real_news_df], ignore_index=True
)
df.to_csv("data/veritas_dataset_test.csv", mode="a")
