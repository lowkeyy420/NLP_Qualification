import numpy as np
import pandas as pd

df = pd.read_csv('./Dataset/cyberbullying_tweets.csv')
df = df.dropna()

positive = df[df['cyberbullying_type'] == "not_cyberbullying"]
negative = df[df['cyberbullying_type'] != "not_cyberbullying"]

positive = positive.drop(['cyberbullying_type'], axis=1)
negative = negative.drop(['cyberbullying_type'], axis=1)

positive = positive.dropna()
negative = negative.dropna()

positive.to_csv('./Dataset/positive.csv', index=False, header=False)
negative.to_csv('./Dataset/negative.csv', index=False, header=False)