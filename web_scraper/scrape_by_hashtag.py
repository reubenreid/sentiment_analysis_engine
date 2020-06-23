import tweepy
import csv
import json


class TwitterScraper(object):

    def __init__(self):
        with open('twitter_credentials.json') as cred_data:
            self.keys = json.load(cred_data)


        with open('hashtag.txt') as hash_data:
            self.hashtag = hash_data.read()


        self.maximum_number_of_tweets_to_be_extracted = 1000
        self.scrape_tweets_by_hashtag()


    def scrape_tweets_by_hashtag(self):
        auth = tweepy.OAuthHandler(self.keys['CONSUMER_KEY'], self.keys['CONSUMER_SECRET'])
        api = tweepy.API(auth)

        with open('tweets_' + self.hashtag + '.csv', 'w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for tweet in tweepy.Cursor(api.search, q='#' + self.hashtag, rpp=100).items(self.maximum_number_of_tweets_to_be_extracted):
                if tweet.user.location is not None:
                    try:
                        writer.writerow([tweet.text.replace('\n', ' ').replace('\r', ''), tweet.user.location])
                    except UnicodeEncodeError:
                        continue
                    except:
                        print('somethings gone very wrong: ' + tweet.text)

        print('Extracted ' + str(self.maximum_number_of_tweets_to_be_extracted) + ' tweets with hashtag #' + self.hashtag)



app = TwitterScraper()