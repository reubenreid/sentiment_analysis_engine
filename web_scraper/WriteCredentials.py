import json

twitter_cred = dict()

twitter_cred['CONSUMER_KEY'] = '37gDZNQZbwfwr99AqMkDZ35jt'
twitter_cred['CONSUMER_SECRET'] = 'p8aJ72M7ZlRfAvv50hj2IL8D9mWBO79KhxI28euaHS7tOfc7Uv'
twitter_cred['ACCESS_KEY'] = '1193840436508602368-8Q0WO9pOvjrQbys7A0XbvrszTNPB2V'
twitter_cred['ACCESS_SECRET'] = 'o7KokHPdUIkNibMxpkEo86eeSEn0IeZQFn271LGiShQQO'

with open('twitter_credentials.json', 'w') as secret_info:
    json.dump(twitter_cred, secret_info, indent=4, sort_keys=True)