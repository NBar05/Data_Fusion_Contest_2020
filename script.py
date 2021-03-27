import numpy as np
import pandas as pd
from scipy import sparse

import pickle
import string

import nltk
from nltk import SnowballStemmer

test = pd.read_parquet('data/task1_test_for_user.parquet')

def preprocess(text):
    text = text.lower()
    text = "".join(c if c not in string.punctuation else f" {c} " for c in text )
    return " ".join(w.strip() for w in text.split())

symbols = ("abcdefghijklmnopqrstuvwyz",
           "абкдефгхижклмнопкрстюввиз")
tr = {ord(a): ord(b) for a, b in zip(*symbols)}

tokenizer = nltk.WhitespaceTokenizer()
# limit_word_cut = lambda sentence: ' '.join([word[:-2] if len(word) > 5 else word for word in tokenizer.tokenize(sentence)])

stemmer = SnowballStemmer("russian")
limit_word_stem = lambda sentence: ' '.join([stemmer.stem(word) for word in tokenizer.tokenize(sentence) if len(word) > 2])

# test['receipt_time'] = test.receipt_time.apply(lambda sentence: sentence[:2]).astype(int)
test['item_name_1'] = test.item_name.apply(preprocess).str.translate(tr).str.replace('x', 'кс')
test['item_name_2'] = test.item_name_1.str.replace('[^\w\s]+', ' ').str.replace('[0-9]+', ' ')
# test['item_name_3'] = test.item_name_2.apply(limit_word_cut)
test['item_name_3'] = test.item_name_2.apply(limit_word_stem)

# ohe = pickle.load(open('ohe', 'rb'))
count0 = pickle.load(open('count0', 'rb'))
count1 = pickle.load(open('count1', 'rb'))
count2 = pickle.load(open('count2', 'rb'))
clf = pickle.load(open('clf_task2', 'rb'))

X_0 = count0.transform(test.item_name)
X_1 = count1.transform(test.item_name_1)
X_2 = count2.transform(test.item_name_3)
# X_3 = ohe.transform(test.loc[:, ['receipt_dayofweek', 'receipt_time', 'item_nds_rate']])

# X = sparse.hstack([X_1, X_2, X_3])
X = sparse.hstack([X_0, X_1, X_2])

pred = clf.predict(X)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id'].values

res[['id', 'pred']].to_csv('answers.csv', index=None)
