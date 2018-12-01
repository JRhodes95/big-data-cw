from __future__ import print_function

import pandas as pd
import numpy as np
import math

import spacy
from nltk import word_tokenize
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


class DeepLearner(object):
    """Convenient wrapper for the Deep Learning approaches in the assignment."""
    def __init__(self):
        print("Implementing deep learning approaches.")

    def get_data(self, file_path):
        """Get data from a CSV file and store as a DataFrame

        Keyword arguments:
        file_path -- the file path of the csv to be read
        """
        self.raw_df = pd.read_csv(file_path)
        print("Read data with shape: ", self.raw_df.shape)


    def to_unicode(self, text):
        """Decode a unicode document."""
        text = text.decode('utf-8')
        return text

    def to_lower(self, text):
        """Convert a text to lowercase chars."""
        text = text.lower()
        return text

    def to_int(self, tokenized_text, lookup):
        """Convert a list of tokens to a list of ints using a lookup dictionary"""
        indexed_text = []
        for token in tokenized_text:
            index = lookup[token]
            indexed_text.append(index)
        return indexed_text

    def feature_extraction(self, tokenized_text):
        """
        Function docstring
        """
        nlp = spacy.load('en_core_web_lg')
        indexed_text = []
        for token in tokenized_text:
            token_object = nlp(token)
            indexed_text.append(token_object.vector)
        return indexed_text

    def split_list(self, a_list, split):
        """Split a list into training and test portions.

        Keyword inputs:
        a_list -- the list to be split
        split -- fraction of the list to be used in the training portion
        """
        split = len(a_list) *(split)
        split_rd = int(math.ceil(split))
        train = a_list[:split_rd]
        test = a_list[split_rd:]
        return  train, test

    def split_format(self, train_frac):
        """
        Outputs a formatted split from an input decimal.

        Example: train_frac = 0.75 -> 75:25
        """
        train_percent = train_frac * 100
        test_percent = 100 - train_percent
        output = "{0}:{1}".format(int(train_percent), int(test_percent))
        return output

    def lstm_approach(self, split=0.6, no_epochs=15):
        """Implements an LSTM to classify texts and returns scores."""
        print("Using a {0} split with {1} epochs.".format(self.split_format(split), no_epochs))
        # Convert stings to unicode and lowercase
        print("Converting text for use in tokenizer.")
        self.raw_df['UNICODE'] = self.raw_df['TEXT'].apply(self.to_unicode)
        self.raw_df['LOWER'] = self.raw_df['UNICODE'].apply(self.to_lower)

        # Tokenize articles
        print("Tokenizing articles.")
        self.raw_df['TOKENIZED'] = self.raw_df['LOWER'].apply(word_tokenize)

        vector_prompt = raw_input("Would you like to use a pretrained word2vec model? (y/n)")
        if 'y' in vector_prompt:
            print("Testing with pretrained word2vec model from spaCy.")
            self.raw_df['TOKEN_VECS'] = self.raw_df['TOKENIZED'].apply(self.feature_extraction)

            x_train, x_test = self.split_list(self.raw_df['TOKEN_VECS'].tolist(), split)
            y_train, y_test = self.split_list(self.raw_df['LABEL'].tolist(), split)

            print("Training samples: {0}, Training labels: {1}".format(len(x_train), len(y_train)))
            print("Test samples: {0}, Test labels: {1}".format(len(x_test), len(y_test)))

            # Pad sequences in training set
            maxlen = 100
            batch_size = 32
            np.random.seed(7)

            print("Padding Sequences for LSTM.")
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

            print('Build model...')
            model = Sequential()
            model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(1, activation='sigmoid'))

            # try using different optimizers and different optimizer configs
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            print('Train...')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=no_epochs,
                      validation_data=(x_test, y_test))
            score, acc = model.evaluate(x_test, y_test,
                                        batch_size=batch_size)

            return (score, acc)

        else:
            print("Creating own text embedding.")
            # Create vocab dictionary
            print("Creating vocab dictionary.")
            tokenised_articles_list = self.raw_df['TOKENIZED'].tolist()
            flattened_articles = [item for sublist in tokenised_articles_list for item in sublist]
            vocab = sorted(list(set(flattened_articles)))
            word_to_int = dict((c,i) for i,c in enumerate(vocab))

            # Convert tokenized articles to article of ints
            print("Converting articles to articles of ints.")
            self.raw_df['TOKEN_INTS'] = self.raw_df['TOKENIZED'].apply(self.to_int, args=(word_to_int,))

            # Split into training and test self.raw_dfs
            print("Splitting into training and test sets.")

            x_train, x_test = self.split_list(self.raw_df['TOKEN_INTS'].tolist(), split)
            y_train, y_test = self.split_list(self.raw_df['LABEL'].tolist(), split)

            print("Training samples: {0}, Training labels: {1}".format(len(x_train), len(y_train)))
            print("Test samples: {0}, Test labels: {1}".format(len(x_test), len(y_test)))

            # Pad sequences in training set
            maxlen = 100
            vocab_len = len(vocab)
            batch_size = 32
            np.random.seed(7)

            print("Padding Sequences for LSTM.")
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

            print('Build model...')
            model = Sequential()
            model.add(Embedding(vocab_len, 128))
            model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
            model.add(Dense(1, activation='sigmoid'))

            # try using different optimizers and different optimizer configs
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            print('Train...')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=no_epochs,
                      validation_data=(x_test, y_test))
            score, acc = model.evaluate(x_test, y_test,
                                        batch_size=batch_size)

            return (score, acc)
