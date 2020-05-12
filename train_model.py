#!/usr/bin/env python
# coding: utf-8

# ### IMPORT MODULES

# Make sure you run requirements.txt to import all the modules

# In[105]:


import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import json
import pickle
import re
from pathlib import Path
import nltk
from time import time
from emoji import demojize
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer


# ### DATASET

# Dataset class defines how the data is handled throught the model

# In[106]:


class Dataset:
    def __init__(self, filename, label_col='label', text_col='text'):
        self.filename = filename
        self.label_col = label_col
        self.text_col = text_col

    def data(self):
        data = self.dataframe[[self.label_col, self.text_col]].copy()
        data.columns = ['label', 'text']
        return data

    def cleaned_data(self):
        data =  self.dataframe[[self.label_col, 'cleaned']]
        data.columns = ['label', 'text']
        return data

    def load(self):
        df = pd.read_csv(Path(self.filename).resolve())
        self.dataframe = df

    def preprocess_texts(self, quiet=False):
        self.dataframe['cleaned'] = preprocess(self.dataframe[self.text_col], quiet)


# ### DATA PREPROCESSING

# 1. Lowercasing the post content
# 2. Remove hyperlinks 
# 3. Converting emojis to text
# 4. Remove stopwords

# In[107]:


def preprocess(texts, quiet=False):
    texts = texts.str.lower()
    texts = texts.str.replace(r"(http|@)\S+", "")
    texts = texts.apply(demojize)
    texts = texts.str.replace(r"::", ": :")
    texts = texts.str.replace(r"’", "'")
    texts = texts.str.replace(r"[^a-z\':_]", " ")
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"n't", ' not')
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    texts = texts.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    print("Preprocessing done")
    return texts


# ### TRAIN FILE

# The JSON formatted train file is converted to CSV format with the columns - id, label and text

# In[108]:


def train_file(f):
    file = open(f)
    data = json.load(file)
    df = pd.DataFrame(columns=["id","label","text"])
    for key in data:
        emotions = data[key]["emotion"]
        for emo,value in emotions.items():
            if(value == True):
                df = df.append({"id":key,"label":emo,"text":data[key]["body"]}, ignore_index = True)
    df.to_csv("nlp_train.csv")
    
#Change the name of the JSON Train file accordingly
train_file("nlp_train.json")


# In[109]:


#Project path is added to the ENV VARIABLE - PYTHONPATH
sys.path.append(Path(os.path.join(os.path.abspath(''), '../')).resolve().as_posix())
dataset_path = Path('nlp_train.csv').resolve()
dataset = Dataset(dataset_path)
dataset.load()
dataset.preprocess_texts()
cleaned_df = dataset.cleaned_data()
cleaned_df.head()


# ### TOKENIZE

# The input text words are converted into word vectors and saved to "tokenizer.pkl" 

# In[110]:


num_words = cleaned_df['text'].str.split().str.len().sum()
tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(cleaned_df.text)
file_to_save = Path('tokenizer.pickle').resolve()
with file_to_save.open('wb') as file:
    pickle.dump(tokenizer, file)


# ### TRAIN, TEST AND VALIDATION DATA

# The test data is also split here

# In[111]:


train = pd.DataFrame(columns=['label', 'text'])
validation = pd.DataFrame(columns=['label', 'text'])
for label in cleaned_df.label.unique():
    label_data = cleaned_df[cleaned_df.label == label]
    train, validation = train_test_split(label_data,test_size=0.2)
    train = pd.concat([train, train_data])
    validation = pd.concat([validation, validation_data])

test.to_csv("nlp_test.csv") #Test data created to test the data


# ### NEURAL MODEL  

# A Stacked Bidirectional LSTM, CNN model is created

# In[112]:


input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = len(cleaned_df.label.unique())
embedding_dim = 500
# input_length = cleaned_df['text'].str.split().str.len().max()
input_length = 100
lstm_units = 128
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout=0.2
filters=64
kernel_size=3

input_layer = Input(shape=(input_length,))
output_layer = Embedding(
  input_dim=input_dim,
  output_dim=embedding_dim,
  input_shape=(input_length,)
)(input_layer)

output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer = Bidirectional(
LSTM(lstm_units, return_sequences=True,
     dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)

output_layer = Bidirectional(
LSTM(lstm_units, return_sequences=True,
     dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)

output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                    kernel_initializer='glorot_uniform')(output_layer)

avg_pool = GlobalAveragePooling1D()(output_layer)
max_pool = GlobalMaxPooling1D()(output_layer)
output_layer = concatenate([avg_pool, max_pool])

output_layer = Dense(num_classes, activation='softmax')(output_layer)

model = Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# ### ENCODING & TRAINING

# In[113]:


train_sequences = [text.split() for text in train.text]
validation_sequences = [text.split() for text in validation.text]
list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)
x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)

encoder = LabelBinarizer()
encoder.fit(cleaned_df.label.unique())

encoder_path = Path('../', 'encoder.pickle')
with encoder_path.open('wb') as file:
    pickle.dump(encoder, file)

y_train = encoder.transform(train.label)
y_validation = encoder.transform(validation.label)


# In[114]:


batch_size = 128
epochs = 20
model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation)
)


# In[115]:


model_file = Path('../model_weights.h5').resolve()
model.save_weights(model_file.as_posix())


# In[116]:


train_acc = model.evaluate(x_train, y_train, verbose = 0)
test_acc = model.evaluate(x_validation, y_validation, verbose = 0)


# In[117]:


print(train_acc, test_acc)


# In[ ]:




