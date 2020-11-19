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

    df = None
    movie_name = 'joker'
    numOfTweets = 5
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


        if j >= 5:
            time.sleep(16*60)
            j = -1
        if i >= numOfTweets or (nt is None):
            break
        i += 1
        j += 1

    df.to_csv(saveFile)
    # using clean function
    data = clean_data(df)

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
    # make array contain all nt  and access them using it with out side counter
    arr_component = [nt, avg_rating]
    # return nt
    return arr_component

if __name__ == '__main__':
    su = 0
    avg =0
    sum =0
    Total_Avg = 0
    i=0
    s = r'/Users/hayoom/Downloads/config_project2/config.yaml'
    s2= r'/Users/hayoom/Downloads/config2.yaml'
    s3 =r'/Users/hayoom/Downloads/config_3.yaml'
    s4 = r'/Users/hayoom/Downloads/config_4.yaml'
    s5 = r'/Users/hayoom/Downloads/config_5.yaml'
    Key_list =[s, s2, s3, s4]
    file_list = ["t1.csv","t2.csv","t3.csv","t4.csv"]
    n0= None
    var =0

    # try to add more config file  !!
    # then try to return the number of total tweeet , and other object ,nig +pos .....
    # try to plot here not in every call
    print(len(Key_list) )
    for i in range(len(Key_list)):
        if( i == 0 ):
            return_result = main(Key_list[i],file_list[i],None)
        else:
            return_result = main(Key_list[i],file_list[i],return_result[0])
        su += return_result[1]
        var= var+1
    # first call take nt = None
    print("sssssss   ",su,"  len(s_list)  :",len(Key_list) )
    avg = su / len(Key_list)
    print("avg ",avg)
    # m= main(s,"tweet1.csv",None,)
    # i+=1
    # sum+= m[1]
    # m2 = main(s2,"tweet2.csv",m[0],)
    # i += 1
    # print(i)
    # sum += m2[1]
    # print("sum = ",sum)
    # Total_Avg = sum/i
    # print("total = ",Total_Avg)
