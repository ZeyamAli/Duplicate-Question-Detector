import numpy as np
import pandas as pd
import sklearn
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import distance
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
import pickle

class Preprocessing:
    def __init__(self):
        # Load the saved CountVectorizer
        with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
            self.cv = pickle.load(vectorizer_file)

        self.STOP_WORDS = set(stopwords.words("english"))

    def preprocess(self, q):
        q = str(q).lower().strip()

        # Replace certain special characters with their string equivalents
        q = q.replace('%', ' percent')
        q = q.replace('$', ' dollar ')
        q = q.replace('₹', ' rupee ')
        q = q.replace('€', ' euro ')
        q = q.replace('@', ' at ')
        q = q.replace('[math]', '')

        # Replace numbers with units
        q = q.replace(',000,000,000 ', 'b ')
        q = q.replace(',000,000 ', 'm ')
        q = q.replace(',000 ', 'k ')
        q = re.sub(r'([0-9]+)000000000', r'\1b', q)
        q = re.sub(r'([0-9]+)000000', r'\1m', q)
        q = re.sub(r'([0-9]+)000', r'\1k', q)

        # Decontracting words
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            # Add more contractions as needed...
        }
        q_decontracted = [contractions[word] if word in contractions else word for word in q.split()]
        q = ' '.join(q_decontracted)

        # Remove HTML tags
        q = BeautifulSoup(q, "html.parser").get_text()

        # Remove punctuations
        q = re.sub(r'\W', ' ', q).strip()

        return q

    def test_common_words(self, q1, q2):
        w1 = set(map(str.strip, q1.lower().split()))
        w2 = set(map(str.strip, q2.lower().split()))
        return len(w1 & w2)

    def test_total_words(self, q1, q2):
        w1 = set(map(str.strip, q1.lower().split()))
        w2 = set(map(str.strip, q2.lower().split()))
        return len(w1) + len(w2)

    def test_fetch_token_features(self, q1, q2):
        SAFE_DIV = 0.0001
        token_features = [0.0] * 8

        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if not q1_tokens or not q2_tokens:
            return token_features

        q1_words = set([word for word in q1_tokens if word not in self.STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in self.STOP_WORDS])
        q1_stops = set([word for word in q1_tokens if word in self.STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in self.STOP_WORDS])

        common_word_count = len(q1_words & q2_words)
        common_stop_count = len(q1_stops & q2_stops)
        common_token_count = len(set(q1_tokens) & set(q2_tokens))

        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])

        return token_features

    def test_fetch_length_features(self, q1, q2):
        length_features = [0.0] * 3

        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if not q1_tokens or not q2_tokens:
            return length_features

        length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
        length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

        strs = list(distance.lcsubstrings(q1, q2))
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0.0

        return length_features

    def test_fetch_fuzzy_features(self, q1, q2):
        fuzzy_features = [0.0] * 4
        fuzzy_features[0] = fuzz.QRatio(q1, q2)
        fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
        fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
        fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
        return fuzzy_features
    
    def query_point_creator(self, q1, q2):
        if not q1 or not q2:
            raise ValueError("Both q1 and q2 must be provided!")

        input_query = []

        q1 = self.preprocess(q1)
        q2 = self.preprocess(q2)

        input_query.append(len(q1))
        input_query.append(len(q2))
        input_query.append(len(q1.split()))
        input_query.append(len(q2.split()))
        input_query.append(self.test_common_words(q1, q2))
        input_query.append(self.test_total_words(q1, q2))
        input_query.append(round(self.test_common_words(q1, q2) / self.test_total_words(q1, q2), 2))

        token_features = self.test_fetch_token_features(q1, q2)
        input_query.extend(token_features)

        length_features = self.test_fetch_length_features(q1, q2)
        input_query.extend(length_features)

        fuzzy_features = self.test_fetch_fuzzy_features(q1, q2)
        input_query.extend(fuzzy_features)

        # Transform using loaded CountVectorizer
        q1_bow = self.cv.transform([q1]).toarray()
        q2_bow = self.cv.transform([q2]).toarray()

        return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))


    # def query_point_creator(self, q1, q2):
    #     input_query = []

    #     q1 = self.preprocess(q1)
    #     q2 = self.preprocess(q2)

    #     input_query.append(len(q1))
    #     input_query.append(len(q2))
    #     input_query.append(len(q1.split()))
    #     input_query.append(len(q2.split()))
    #     input_query.append(self.test_common_words(q1, q2))
    #     input_query.append(self.test_total_words(q1, q2))
    #     input_query.append(round(self.test_common_words(q1, q2) / self.test_total_words(q1, q2), 2))

    #     token_features = self.test_fetch_token_features(q1, q2)
    #     input_query.extend(token_features)

    #     length_features = self.test_fetch_length_features(q1, q2)
    #     input_query.extend(length_features)

    #     fuzzy_features = self.test_fetch_fuzzy_features(q1, q2)
    #     input_query.extend(fuzzy_features)

    #     q1_bow = self.cv.transform([q1]).toarray()
    #     q2_bow = self.cv.transform([q2]).toarray()

    #     return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))
