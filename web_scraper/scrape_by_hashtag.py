import tweepy
import csv
import json


class TwitterScraper(object):

    def __init__(self):
        with open('twitter_credentials.json') as cred_data:
            self.keys = json.load(cred_data)

        self.scrape_tweets_by_hashtag()

    def scrape_tweets_by_hashtag(self):
        auth = tweepy.OAuthHandler(self.keys['CONSUMER_KEY'], self.keys['CONSUMER_SECRET'])
        api = tweepy.API(auth)

        maximum_number_of_tweets_to_be_extracted = 1000

        # hashtag = input('Enter the hashtag you want to scrape- ')
        hashtag = 'python'

        for tweet in tweepy.Cursor(api.search, q='#' + hashtag, rpp=100).items(maximum_number_of_tweets_to_be_extracted):
            if tweet.geo is not None:
                print('has geo')
                with open('raw_tweets_with_hashtag_' + hashtag + '.csv', 'w') as out_file:
                    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([tweet.text, tweet.geo])

        print('Extracted ' + str(maximum_number_of_tweets_to_be_extracted) + ' tweets with hashtag #' + hashtag)


        ## download and parse into json in separate steps

app = TwitterScraper()

#with open('tweets_with_hashtag_' + hashtag + '.json', 'a') as out_file:
# zjson.dump(tweet._json, out_file, indent=4, sort_keys=True)