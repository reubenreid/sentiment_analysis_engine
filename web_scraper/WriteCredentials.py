import json

twitter_cred = dict()

twitter_cred['CONSUMER_KEY'] = ''
twitter_cred['CONSUMER_SECRET'] = ''
twitter_cred['ACCESS_KEY'] = ''
twitter_cred['ACCESS_SECRET'] = ''

with open('twitter_credentials.json', 'w') as secret_info:
    json.dump(twitter_cred, secret_info, indent=4, sort_keys=True)