import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from LAB4.naive_bayes_lab import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np

# load data
df = pd.read_csv('labeledTrainData.tsv', sep='\t', index_col=False)
print(df.head(10))
cols = df.columns
print(cols)

stops = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def tokenizer_func(sample):
    words = nltk.word_tokenize(sample)

    words = [word for word in words if word.isalnum()]
    words = [word for word in words if not word.isdigit()]

    words = [w for w in words if not w in stops]

    word_list = [lemmatizer.lemmatize(w) for w in words]

    return word_list

def preprocessor_func(sample):
    no_html_words = BeautifulSoup(sample, features="html.parser").get_text()

    lower_words = no_html_words.lower()

    return lower_words

vectorizer = CountVectorizer(analyzer = "word", tokenizer = tokenizer_func, preprocessor = preprocessor_func, stop_words = None, max_features = 5000)

# binary features from words

corpus = df['review'].values
t= df['sentiment'].values

X_tr, X_te, t_tr, t_te =  train_test_split(corpus, t, test_size=0.2, random_state=0)

X_tr = vectorizer.fit_transform(X_tr).toarray()

print(vectorizer.get_feature_names_out())

#TODO Train with the BernoulliNB class from scikit-learn

#TODO Evaluate the model on the training set

print("Accuracy on the training set: ", accuracy)

#TODO Train with our custom implementation of Naive Bayes and evaluate on the training set
print("Accuracy on the training set with your implementation: ", accuracy)


X_te = vectorizer.transform(X_te).toarray()
N_te = X_te.shape[0]

#Evaluate the model trained with our custom Naive Bayes implementation on the test set
print("Accuracy on the test set: ", accuracy)

# Now test with your sentences
test_corpus = ['A very bad movie',
               'This is very nice']
X_te_new = vectorizer.transform(test_corpus).toarray()

t_hat_te_cnb = custom_classifier.predict_v(X_te_new)
print(t_hat_te_cnb)
