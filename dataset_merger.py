import os
import json
from datetime import date

root_path = "./dataset/bbc"

def merge():
    years = os.listdir(root_path)
    
    articles = []

    for year in years:
        year_path = os.path.join(root_path, year)
        
        articles += merge_year(year_path)
    
    return articles

def merge_year(year_path):
    months = os.listdir(year_path)

    articles = []
    
    for month in months:
        month_path = os.path.join(year_path, month)
    
        articles += merge_month(month_path)

    return articles

def merge_month(month_path):
    days = os.listdir(month_path)
    
    articles = []

    for day in days:
        day_path = os.path.join(month_path, day)
        
        articles += get_articles(day_path)
    
    return articles

def get_articles(day_path):
    path = os.path.join(day_path, "articles")

    articles = []

    try:
        with open(path) as f:
            data = json.load(f)
            for article in data["articles"]:
                article["content"] = str(article["content"]).replace("\n", " ").replace(",", "")
                articles.append(article)
    except:
        pass

    return articles


def print_to_csv(articles):
    import pandas as pd
    articles = json.dumps(articles) 

    df = pd.read_json(articles)
    df.to_csv('dataset.csv')


if __name__ == "__main__":
    articles = merge()

    print_to_csv(articles)