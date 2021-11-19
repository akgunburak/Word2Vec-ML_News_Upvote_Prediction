import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import table
from keras.models import Sequential
from keras.layers import Dense


def load_data(path, batch_size):
    """
    
    Parameters
    ----------
    path : str
        Path of dataset
    batch_size : int
        The number of sample to be fetched randomly

    Returns
    -------
    dataset : tf.dataset

    """
    dataset = tf.data.experimental.make_csv_dataset(path,
                                                    batch_size=batch_size)
    return dataset


# Loading the sample data
dataset = load_data("data/Eluvio_DS_Challenge.csv", 500000)


def text_preprocessing(dataset):
    """

    Parameters
    ----------
    dataset : tf.dataset
        Dataset that contains the features

    Returns
    -------
    dataset : Pandas DataFrame
        Dataframe with preprocessed text columns

    """
    [(dataset)] = dataset.take(1)
    dataset = pd.DataFrame(dataset)
    stop_words = stopwords.words('english')
    
    for column in dataset:
        if column == "title":
            dataset[column] = dataset[column].map(lambda x: x.decode().lower())
            dataset[column] = dataset[column].map(lambda x: nltk.word_tokenize(x))  # Tokenization of words
            dataset[column] = dataset[column].map(lambda x: [word for word in x if word.isalnum()])  # Getting only the alphabet letters (a-z) and numbers (0-9) 
            dataset[column] = dataset[column].map(lambda x: [word for word in x if not word in stop_words])  # Removing stop words
    return dataset


# Preprocessing the text data
dataset = text_preprocessing(dataset)


# load the Google's word2vec model
pre_trained_file = 'data/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(pre_trained_file, binary=True)


def feature_engineering(dataset):
    """

    Parameters
    ----------
    dataset : Pandas DataFrame

    Returns
    -------
    dataset : Pandas DataFrame
        Dataset that all columns are numeric

    """

    # Calculating the average of vectors for each row in the title column
    dataset["title"] = dataset["title"].map(lambda x: [word for word in x if word in model.vocab])
    dataset["vector_average"] = dataset["title"].map(lambda row: np.mean([model[word] for word in row], axis=0))
    dataset.dropna(inplace=True)
    vector_average = pd.DataFrame(np.array([np.array(row) for row in dataset["vector_average"]]))
    
    # Adding up votes column
    vector_average["up_votes"] = dataset["up_votes"][dataset["up_votes"].between(dataset["up_votes"].quantile(.15), dataset["up_votes"].quantile(.85))] # without outliers
    vector_average.dropna(inplace=True)
    return vector_average


# Implementing feature engineering
dataset = feature_engineering(dataset)


def split_train_test(dataset):
    """

    Parameters
    ----------
    dataset : Pandas DataFrame

    Returns
    -------
    X_train : Numpy Array
        
    X_test : Numpy Array
        
    y_train : Numpy Array
        
    y_test : Numpy Array

    """
    # Determining X values
    X = np.array(dataset.drop(['up_votes'], 1))
    
    # Determining y values
    y = np.array(dataset["up_votes"])
    
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = split_train_test(dataset)


# Building neural network model
def neural_network():
    model = Sequential()
    
    # Iput layer
    model.add(Dense(128, kernel_initializer='normal',
                    input_dim=X_train.shape[1],
                    activation='relu'))
    
    # Hidden layers
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    
    # Output Layer
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    
    # Compiling the network
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mean_squared_error'])
    return model


def fit_regressors(X_train, X_test, y_train, y_test):
    """

    Parameters
    ----------
    X_train : Numpy Array
        
    X_test : Numpy Array
        
    y_train : Numpy Array
        
    y_test : Numpy Array

    Returns
    -------
    scores : Pandas DataFrame
        Root Mean Squared Error of each algorithm

    """
    # Adding sklearn classifiers into a list
    regressors = []
    
    model1 = LinearRegression()
    regressors.append(model1)
    model2 = tree.DecisionTreeRegressor()
    regressors.append(model2)
    model3 = RandomForestRegressor()
    regressors.append(model3)
    model4 = xgboost.XGBRegressor()
    regressors.append(model4)
    model5 = KNeighborsRegressor()
    regressors.append(model5)
    model6 = neural_network()
    regressors.append(model6)


    clfs = ["LinReg", "DT", "RF", "XGB", "KNN", "NN"]
    scores = pd.DataFrame(index=clfs, columns=["RMSE"])
    
    # Fitting the machine learning algorithms
    row = 0
    for clf in regressors:
        if row == len(regressors)-1:
            clf.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split = 0.2)  # Fitting the Neural Network
            y_pred = clf.predict(X_test) 
        else:
            clf.fit(X_train, y_train)  # Fitting ML regressors
            y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**(0.5)
        scores["RMSE"][row] = round(rmse, 4)
        row += 1
    return(scores)


# Fitting the classifiers and calculating the accuracies
scores = fit_regressors(X_train, X_test, y_train, y_test)


# Plotting the results
fig, ax = plt.subplots(1, 1)
table(ax, np.round(scores), loc="upper right", colWidths=[0.15, 0.15, 0.15])
ax.set_ylabel('RMSE')
ax.set_xlabel('Algorithms')
ax.set_title('Performances of Regressors')
scores.plot(ax=ax, kind="bar",
            figsize=(15, 10),
            ylim=(0, (max(scores["RMSE"])+5)),
            legend=None)
plt.savefig('results/regression_pt_results.png')