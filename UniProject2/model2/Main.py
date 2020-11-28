import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from model2.pred import read_list_of_words, tfidf_transform, replace_tfidf_words, make_predictions
from model2.preprocessing import clean_data, create_sentences


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
    # print(url)
    response = requests.request("GET", url, headers=headers)
    print(response.json()["meta"])

    for tweet in response.json()["data"]:
        tweets.append(tweet)
    try:
        nt = response.json()["meta"]["next_token"]
        return tweets, nt
    except KeyError as k:
        print(str(k))
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
    labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
              'Negative [' + str(negative) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'gold', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title('How people think of ' + searchTerm.upper() + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main(fileURL, saveFile, nt):
    df = None
    movie_name = 'spiderman'
    numOfTweets = 4
    numOfTweets = 3 if numOfTweets >= 4 else numOfTweets - 1
    # i removed the movie string that was give a less number of data
    query = f'-is:retweet lang:en ("{movie_name}" )'  # Set rules here
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
            time.sleep(16 * 60)
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

    positive_words = read_list_of_words('../sentiment dictionary/Positive.txt')
    positive_dict = dict(zip(positive_words, np.ones(len(positive_words))))
    negative_words = read_list_of_words('../sentiment dictionary/Negative.txt')
    negative_dict = dict(zip(negative_words, -1 * np.ones(len(positive_words))))

    sentiment_dict = {**negative_dict, **positive_dict}

    replaced_tfidf_words = weights.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)

    final_df = make_predictions(weights, sentiment_dict, replaced_tfidf_words)

    positive = final_df[final_df['prediction'] == 1]['prediction'].count()
    neutral = final_df[final_df['prediction'] == 0]['prediction'].count()
    negative = final_df[final_df['prediction'] == -1]['prediction'].count()
    component_arr = [nt, positive, neutral, negative, total_count, movie_name]

    return component_arr


if __name__ == '__main__':
    sum = 0
    Total_Avg = 0
    Total_positive = 0
    Total_neutral = 0
    Total_negative = 0
    Total_count = 0
    searchTerm =None

    s1 = r'/Users/hayoom/Downloads/config_project2/config.yaml'
    s2 = r'/Users/hayoom/Downloads/config2.yaml'
    s3 = r'/Users/hayoom/Downloads/config_3.yaml'
    s4 = r'/Users/hayoom/Downloads/config_4.yaml'
    s5 = r'/Users/hayoom/Downloads/config_5.yaml'
    s6 = r'/Users/hayoom/Downloads/config_6.yaml'
    s7 = r'/Users/hayoom/Downloads/config_7.yaml'
    s8 = r'/Users/hayoom/Downloads/config_8.yaml'
    s9 = r'/Users/hayoom/Downloads/config_9.yaml'
    s10 = r'/Users/hayoom/Downloads/config_10.yaml'
    s11 = r'/Users/hayoom/Downloads/config_11.yaml'
    Key_list = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    file_list = ["tweetsData1.csv", "tweetsData2.csv", "tweetsData3.csv", "tweetsData4.csv", "tweetsData5.csv",
                 "tweetsData6.csv", "tweetsData7.csv", "tweetsData8.csv", "tweetsData9.csv", "tweetsData10.csv",
                 "tweetsData11.csv"]
    # i didn't use threads because every bearer ky depend on the last token the previous one return , to give different data
    for counter in range(len(Key_list)):
        if counter == 0:
            return_result = main(Key_list[counter], file_list[counter], None)
        else:
            return_result = main(Key_list[counter], file_list[counter], return_result[0])
        # checking if the model collect all data
        if return_result[0] is None:
            break
        Total_positive += return_result[1]
        Total_neutral += return_result[2]
        Total_negative += return_result[3]
        Total_count += return_result[4]
    searchTerm = return_result[5]
    print( "  length of bearer tokens list  :", len(Key_list))
    positive_percentage = (Total_positive / Total_count) * 100
    neutral_percentage = (Total_neutral / Total_count) * 100
    negative_percentage = (Total_negative / Total_count) * 100

    avg_rating = (Total_positive * 10 + Total_negative * 0 + (Total_neutral + 1) * 5) / Total_count
    print(f'Average Rating: {avg_rating} / 10', 'total number of tweets :', Total_count)
    plotPieChart(Total_positive, Total_neutral, Total_negative, searchTerm, Total_count)
    #     # # initilize thread 1 for array1Calc function
    #     # t1 = threading.Thread(target=array1Calc, args=(1,), daemon=True)
    #     # # initilize thread 1 for array1Calc function
    #     # t2 = threading.Thread(target=array2Calc, args=(2,), daemon=True)
    #     # # start threads
    #     # t1.start()
    #     # t2.start()
    #     # # force the programme to wait for threads until they finish
    #     # t1.join()
    #     # t2.join()
