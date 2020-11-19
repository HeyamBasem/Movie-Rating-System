import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from model2.pred import read_list_of_words, tfidf_transform, replace_tfidf_words, make_predictions
from model2.preprocessing import  clean_data, create_sentences ,text_process
import threading


def create_twitter_url(query, next_token):
    max_results = 100
    tweet_fields = "id,text,created_at,public_metrics,source"
    mrf = f"max_results={max_results}"
    q = f"query={query}"
    tf = f"tweet.fields={tweet_fields}"
    if next_token is None:
        url = f"https://api.twitter.com/2/tweets/search/recent?{mrf}&{q}&{tf}"
    else:
        nt = f"next_token={next_token}"
        url = f"https://api.twitter.com/2/tweets/search/recent?{mrf}&{q}&{tf}&{nt}"
    return url


def process_yaml(fileURL):
    with open(fileURL) as file:
        return yaml.safe_load(file)


def create_bearer_token(data):
    return data["search_tweets_api"]["bearer_token"]


def twitter_auth_and_connect(bearer_token, url):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers


def get_tweets(url, headers):
    tweets = []
    responce = requests.request("GET", url, headers=headers)
    # We can use this if we want to write the data to a file...But it's not working... Something is wrong and I don't know what it is
    # with open('Tweets.json', 'a') as file:
    #     file.write(responce)
    print(responce.json())
    for tweet in responce.json()["data"]:
        tweets.append(tweet)
    try:
        nt = responce.json()["meta"]["next_token"]
        return tweets, nt
    except():
        return tweets, None


def create_update_dataframe(tweets, df=None):
    temp = pd.DataFrame([tweet["text"] for tweet in tweets], columns=['Tweet'])
    temp['len'] = np.array([len(tweet["text"]) for tweet in tweets])
    temp['metrics'] = np.array([tweet["public_metrics"] for tweet in tweets])
    temp['source'] = np.array([tweet["source"] for tweet in tweets])
    if df is None:
        return temp
    else:
        return df.append(temp, ignore_index=True)


def plotPieChart(positive, negative, neutral, searchTerm, noOfSearchTerms):
    labels = ['Positive [' + str(positive) + '%]',  'Neutral [' + str(neutral) + '%]',
              'Negative [' + str(negative) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'gold', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title('How people think of ' + searchTerm.upper() + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main(fileURL, saveFile ,nt):
    # nt = None
    df = None
    movie_name = 'joker'
    numOfTweets = 4
    numOfTweets = 3 if numOfTweets >= 4 else numOfTweets - 1
    query = f'-is:retweet lang:en ("{movie_name}" movie)'  # Set rules here
    data = process_yaml(fileURL)
    bearer_token = create_bearer_token(data)

    i = 0
    j = 0

    while (i == 0) or (nt is not None):
        url = create_twitter_url(query, next_token=nt)
        headers = twitter_auth_and_connect(bearer_token, url)
        tweets, nt = get_tweets(url, headers=headers)
        df = create_update_dataframe(tweets, df)


        if j >= 4:
            time.sleep(16*60)
            j = -1
        if i >= numOfTweets or (nt is None):
            break
        i += 1
        j += 1

    df.to_csv(saveFile)
    # using clean function
    data = clean_data(df)
    # using text_process func
    # data = text_process(df)
    # data['Tweet'] = df['Tweet'].apply(text_process)
    #  text process using replace emoji function
    create_sentences(data)
    total_count = data['Tweet'].count()
    weights = data.copy()
    features, transformed = tfidf_transform(weights)

    positive_words = read_list_of_words('/Users/hayoom/Downloads/SentimentAnalysis-master 2/Mark 1/Positive.txt')
    positive_dict = dict(zip(positive_words, np.ones(len(positive_words))))

    negative_words = read_list_of_words('/Users/hayoom/Downloads/SentimentAnalysis-master 2/Mark 1/Negative.txt')
    negative_dict = dict(zip(negative_words, -1 * np.ones(len(positive_words))))

    sentiment_dict = {**negative_dict, **positive_dict}

    replaced_tfidf_words = weights.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)

    final_df = make_predictions(weights, sentiment_dict, replaced_tfidf_words)

    positive = final_df[final_df['prediction'] == 1]['prediction'].count()
    neutral = final_df[final_df['prediction'] == 0]['prediction'].count()
    negative = final_df[final_df['prediction'] == -1]['prediction'].count()

    positive_percentage = (positive / total_count) * 100
    neutral_percentage = (neutral / total_count) * 100
    negative_percentage = (negative / total_count) * 100

    avg_rating = (positive *10 + negative * 0 + (neutral +1)* 5)/total_count
    print(f'Average Rating: {avg_rating} / 10')
    print(nt)
    return nt
    # make a function heyam with only comemted rows
    # plotPieChart(positive_percentage, negative_percentage, neutral_percentage, movie_name, total_count)
    # labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
    #           'Negative [' + str(negative) + '%]']
    # colors = ['yellowgreen', 'gold', 'red']
    # sizes = [positive_percentage, neutral_percentage, negative_percentage]
    # patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    # plt.legend(patches, labels, loc="best")
    # plt.title('How people are reacting on ' + movie_name + ' by analyzing ' + str(total_count) + ' Tweets.')
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    # main()
    # initilize thread 1 for array1Calc function
    s = r'/Users/hayoom/Downloads/config_project2/config.yaml'
    s2= r'/Users/hayoom/Downloads/config2.yaml'
    # s3 = ''
    # s2 = r'‎/Desktop/project2/config6.yaml'
    # s3 = r'/Users/hayoom/Downloads/config_project2/config3.yaml'
    # error in s3 path !!!
    n1='b26v89c19zqg8o3fosbv32ymkum6e5c1cyq6g4egpy8ot'

    nn = 'b26v89c19zqg8o3fosbvicux7dhx1zjz2equtkkswrx4t'
    m= main(s,"tweet1.csv",None,)
    n = "b26v89c19zqg8o3fosbv32ybwzja5jcmlpkklddih44n1"
    main(s2,"tweet2.csv",m,)
    # next_token
    # nt = responce.json()["meta"]["next_token"]
    # KeyError: 'next_token'
    # type error occurs when the nt is static so we must return it as above in the method

    # how to synchronize a single resource between threads.
    # t1 = threading.Thread(target=main, args=(s,"tweet1.csv",None,), daemon=True)
    # n = "b26v89c19zqg8o3fosbv32ybwzja5jcmlpkklddih44n1"
    # t2 = threading.Thread(target=main, args=(s2,"tweet2.csv",n,), daemon=True)

    # t3 = threading.Thread(target=main, args=(r'‎/iCloud Drive/Desktop/project2/config3.yaml'), daemon=True)

    # start threads
    # t1.start()
    print("reach here")
    # t2.start()
    # t3.start()

    # force the programme to wait for threads until they finish
    # t1.join()
    # t2.join()
    print("reach here 55555555")

    # t3.join()
