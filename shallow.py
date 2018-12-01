from __future__ import print_function

import pandas as pd
import math
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class LemmaTokenizer(object):
    """Tokenizer that reduces a document to lemmas."""
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class ShallowLearner(object):
    """Convenient wrapper for the shallow learning approaches in the assignment."""
    def __init__(self):
        # Something
        print("Shallow learning approaches")

    def get_data(self, file_path):
        """Get data from a CSV file and store as a DataFrame

        Keyword arguments:
        file_path -- the file path of the csv to be read
        """
        self.raw_df = pd.read_csv(file_path)
        print("Read data with shape: ", self.raw_df.shape)


    def score(self, predicted, actual):
        """
        Method to score the effectiveness of the classifier.

        Keyword inputs:
        predicted --    list of the data labels predicted by the classifer
        actual --       list of the data labels provided in the dataset

        Outputs:
        results -   pandas Series object with entries for: accuracy, precision,
                    recall, f1 measure.
        """
        # Calculate accuracy
        accuracy = np.mean(predicted == actual)

        # Count true and false positives and negatives
        tp = tn = fp = fn = 0.0
        for i in range(0, len(actual)):
            if predicted[i] == actual[i] == 1:
                tp +=1
            elif predicted[i] == actual[i] == 0:
                tn +=1
            elif predicted[i] != actual[i] and predicted[i] == 1:
                fp +=1
            elif predicted[i] != actual[i] and predicted[i] == 0:
                fn +=1
        #print("TP:{0}, TN: {1}, FP:{2}, FN:{3}".format(tp,tn,fp,fn))

        #Calculate precision
        try:
            precision = tp/(tp+fp)
        except:
            return("Div by zero error in precision calculation.")

        # Calculate Recall
        try:
            recall = tp / (tp + fn)
        except:
            return("Div by zero error in recall calculation.")

        # Calculate F1 Measure
        try:
            f1 = 2*((precision*recall)/(precision+recall))
        except:
            return("Error in F1 calculation, check for precision and recall errors.")

        # Gather the results
        results = pd.Series()
        results['Accuracy'] = accuracy
        results['Precision'] = precision
        results['Recall'] = recall
        results['F1'] = f1

        return results



    def split_format(self, train_frac):
        """
        Outputs a formatted split from an input decimal.

        Example: train_frac = 0.75 -> 75:25
        """
        train_percent = train_frac * 100
        test_percent = 100 - train_percent
        output = "{0}:{1}".format(int(train_percent), int(test_percent))
        return output

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

    def first_approach(self, split=0.6, no_features=None):
        """Implements the first approach to building a shallow learning classifier.

        Keyword inputs:
        split -- (default 0.6) fraction of the list to be used in the training portion.
        no_features -- (default None) maximum number of features to include in the classifer.

        Implements a TF based classifier on tokenized data with no lemmatizer.
        """
        print("Testing first shallow approach.")
        print("Using a {0} split with a maximum {1} features.".format(self.split_format(split), no_features))

        start_time = time.time()
        vectorizer = CountVectorizer(max_features=no_features)

        # Split the data
        train_data_text, test_data_text = self.split_list(self.raw_df['TEXT'].tolist(), split)
        train_data_label, test_data_label = self.split_list(self.raw_df['LABEL'].tolist(), split)

        # Transform the training and test data to numeric forms
        X1_train = vectorizer.fit_transform(train_data_text)
        X1_test = vectorizer.transform(test_data_text)

        # Fit the MultinomialNB model to the training set
        clf = MultinomialNB().fit(X1_train, train_data_label)
        # Predict values for the test set using the trained classifier
        predicted = clf.predict(X1_test)

        # Score the results
        results = self.score(predicted, test_data_label)
        fin_time = time.time()
        duration = fin_time - start_time

        print("First approach completed in {0} seconds.".format(duration))
        return results

    def second_approach(self, split=0.6, no_features=None):
        """Implements the second approach to building a shallow learning classifier.

        Keyword inputs:
        split -- (default 0.6) fraction of the list to be used in the training portion.
        no_features -- (default None) maximum number of features to include in the classifer.

        Implements a TF based classifier on tokenized data with a lemmatizer.
        """
        print("Testing second shallow approach.")
        print("Using a {0} split with a maximum {1} features.".format(self.split_format(split), no_features))

        start_time = time.time()
        vectorizer = CountVectorizer(
                                    tokenizer=LemmaTokenizer(),
                                    stop_words=stopwords.words('english'),
                                    max_features=no_features
                                    )

        # Split the data
        train_data_text, test_data_text = self.split_list(self.raw_df['TEXT'].tolist(), split)
        train_data_label, test_data_label = self.split_list(self.raw_df['LABEL'].tolist(), split)

        # Transform the training and test data to numeric forms
        X1_train = vectorizer.fit_transform(train_data_text)
        X1_test = vectorizer.transform(test_data_text)

        # Fit the MultinomialNB model to the training set
        clf = MultinomialNB().fit(X1_train, train_data_label)
        # Predict values for the test set using the trained classifier
        predicted = clf.predict(X1_test)

        # Score the results
        results = self.score(predicted, test_data_label)
        fin_time = time.time()
        duration = fin_time - start_time

        print("Second approach completed in {0} seconds.".format(duration))
        return results

    def third_approach(self, split=0.6, no_features=None):
        """Implements the third approach to building a shallow learning classifier.

        Keyword inputs:
        split -- (default 0.6) fraction of the list to be used in the training portion.
        no_features -- (default None) maximum number of features to include in the classifer.

        Implements a TF-IDF based classifier on tokenized data with a lemmatizer.
        """
        print("Testing third shallow approach.")
        print("Using a {0} split with a maximum {1} features.".format(self.split_format(split), no_features))

        start_time = time.time()
        vectorizer = TfidfVectorizer(
                                    tokenizer=LemmaTokenizer(),
                                    stop_words=stopwords.words('english'),
                                    max_features=no_features
                                    )

        # Split the data
        train_data_text, test_data_text = self.split_list(self.raw_df['TEXT'].tolist(), split)
        train_data_label, test_data_label = self.split_list(self.raw_df['LABEL'].tolist(), split)

        # Transform the training and test data to numeric forms
        X1_train = vectorizer.fit_transform(train_data_text)
        X1_test = vectorizer.transform(test_data_text)

        # Fit the MultinomialNB model to the training set
        clf = MultinomialNB().fit(X1_train, train_data_label)
        # Predict values for the test set using the trained classifier
        predicted = clf.predict(X1_test)

        # Score the results
        results = self.score(predicted, test_data_label)
        fin_time = time.time()
        duration = fin_time - start_time

        print("Third approach completed in {0} seconds.".format(duration))
        return results

    def fourth_approach(self, n_range=(2,2), split=0.6, no_features=None):
        """Implements the fourth approach to building a shallow learning classifier.

        Keyword inputs:
        n_range -- (default (2,2)) range of n grams to include, tuple (min_n, max_n)
        split -- (default 0.6) fraction of the list to be used in the training portion.
        no_features -- (default None) maximum number of features to include in the classifer.

        Implements an n-gram based classifier on tokenized data with a lemmatizer.
        """
        print("Testing fourth shallow approach.")
        print("Using a {0} split with a maximum {1} features.".format(self.split_format(split), no_features))

        start_time = time.time()
        vectorizer = CountVectorizer(
                                    tokenizer=LemmaTokenizer(),
                                    stop_words=stopwords.words('english'),
                                    max_features=no_features,
                                    analyzer='word',
                                    ngram_range=n_range
                                    )

        # Split the data
        train_data_text, test_data_text = self.split_list(self.raw_df['TEXT'].tolist(), split)
        train_data_label, test_data_label = self.split_list(self.raw_df['LABEL'].tolist(), split)

        # Transform the training and test data to numeric forms
        X1_train = vectorizer.fit_transform(train_data_text)
        X1_test = vectorizer.transform(test_data_text)

        # Fit the MultinomialNB model to the training set
        clf = MultinomialNB().fit(X1_train, train_data_label)
        # Predict values for the test set using the trained classifier
        predicted = clf.predict(X1_test)

        # Score the results
        results = self.score(predicted, test_data_label)
        fin_time = time.time()
        duration = fin_time - start_time

        print("Fourth approach completed in {0} seconds.".format(duration))
        return results
