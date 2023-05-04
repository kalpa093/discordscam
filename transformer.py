# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.layers import Layer
# %matplotlib inline

from nltk.tokenize import TweetTokenizer
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Sequence
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.layers import InputSpec
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import re

import nltk
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


train_df = pd.read_csv("../discordscam.csv",encoding='cp949')

max_features = 300000
max_features_2 = 100000
max_features_3 = 50000
maxlen = 500
maxlen_2 = 500
maxlen_3 = 500
train_X = train_df["chatting"].fillna("_##_").values
train_X_name = train_df["nickname"].fillna("_##_").values
train_X_url = train_df["url"].fillna("_##_").values


## Tokenize
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)

tokenizer = Tokenizer(num_words=max_features_2)
tokenizer.fit_on_texts(list(train_X_name))
train_X_name = tokenizer.texts_to_sequences(train_X_name)

tokenizer = Tokenizer(num_words=max_features_3)
tokenizer.fit_on_texts(list(train_X_url))
train_X_url = tokenizer.texts_to_sequences(train_X_url)




## Pad
train_X = pad_sequences(train_X, maxlen=maxlen)

train_X_name = pad_sequences(train_X, maxlen=maxlen_2)
train_X_url = pad_sequences(train_X, maxlen=maxlen_3)

train_y = train_df['label'].values


## Split
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.4, random_state=2023)
test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5, random_state=2023)

train_y = train_df['label'].values
train_X_name, test_X_name, train_y, test_y = train_test_split(train_X_name, train_y, test_size=0.4, random_state=2023)
test_X_name, val_X_name, test_y, val_y = train_test_split(test_X_name, test_y, test_size=0.5, random_state=2023)

train_y = train_df['label'].values
train_X_url, test_X_url, train_y, test_y = train_test_split(train_X_url, train_y, test_size=0.4, random_state=2023)
test_X_url, val_X_url, test_y, val_y = train_test_split(test_X_url, test_y, test_size=0.5, random_state=2023)

## shuffling
np.random.seed(2023)
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))

train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]
train_X_name = train_X[trn_idx]
val_X_name = val_X[val_idx]
train_X_url = train_X[trn_idx]
val_X_url = val_X[val_idx]




class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim # d_model
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs = self.dense(concat_attention)
        return outputs

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation="relu"),
             tf.keras.layers.Dense(embedding_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs) # 첫번째 서브층 : 멀티 헤드 어텐션
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # Add & Norm
        ffn_output = self.ffn(out1) # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # Add & Norm


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, max_features, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(max_features, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(maxlen, embedding_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_clf_eval(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1score = 2/(1/precision+1/recall)
    print('Accuracy : {:.4f}\nPrecision : {:.4f}\nRecall : {:.4f}\nf1 score : {:.4f}'.format(accuracy, precision, recall, f1score))

