import pandas as pd

df = pd.read_csv (r'C:\Users\Reuben\Documents\GitHub\sentiment_ui\src\data\uk_online_data_full.csv')
del  df['Tweet content']

with open('C:\\Users\Reuben\Documents\GitHub\sentiment_ui\src\data\\uk_online_data_full_conv.json', 'w') as f:
    for index, row in df.iterrows():
        f.write('  [{},{},{}],\n'.format(row['Longitude'], row['Latitude'], row['Sentiment']))


# x = df.to_string(header=False,
#                   index=False,
#                   index_names=False).split('\n')
# vals = [','.join(ele.split()) for ele in x]
# print(vals)

#df.loc[:, 'Latitude':'Longitude'].to_json (r'C:\Users\Reuben\Documents\GitHub\sentiment_ui\src\data\uk_online_data_full.json')