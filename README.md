# Predicting the Up Votes of News Using Natural Language Processing
&nbsp; The aim of this project is to find the best machine learning technique that predicts the 'Up Votes' of the news. In that regard, I approached the problem as a regression problem. I also investigated whether training a Word2Vec model or using pre-trained word vectors enhances the performance of algorithms. After getting word vectors, I used them as a feature to train and evaluate various ML algorithms.

&nbsp;

## Technologies Used
- **Language and version:** Python - 3.8.8
- **Packages:** Pandas, Numpy, Tensorflow, Keras, Scikit-Learn, Xgboost, NLTK, Gensim, Matplotlib

&nbsp;

## Project Architecture
**1- Loading the Data:** Because the dataset is assumed as large that does not fit into our RAM, I implemented the analyzes on a sample that took randomly from the original dataset. I chose sample size 500.000. I assume that randomly chosen samples represent the population's characteristics.  

**2- Exploratory Data Analysis:** I did a very quick EDA on the sample in order to grasp the structure of the data and decide which operations should I do or which columns are informative.  

**3- Preprocessing the Text Data:** This step includes operations such as: converting all letters to lower, tokenization, removing punctuations and stop words.  

**4- Selecting Word2Vec Model:** After text preprocessing, I define the Word2Vec model here. I investigate two methods: train my own model and use a pre-trained word vectors. Pre-trained one is trained on Google News Dataset. This vectors can be downloaded [**here**](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)  

**5- Feature Engineering:** In this step, I converted the words into 300-dimensional vectors and took the average of them.

**6- Splitting the Data:** In order to obtain train and test sets, I used train_test_split method. I set the test size to 20.  

**7- Building Neural Network:** Before the final step, I built a Neural Network with two hidden layers, each hidden layer containing 256 nodes.  

**8- Fitting and Training the Algorithms:** In the last step, I trained and evaluated the chosen algorithms, then plotted the graphs of result to make comparisons easily.

&nbsp;

## Results
* Word2Vec Model Trained on Our Dataset
![alt text](https://github.com/akgunburak/Word2Vec-ML_News_Upvote_Prediction/blob/master/results/regression_wv_results.png)

* Google's Pre-Trained Word2Vec Model
![alt text](https://github.com/akgunburak/Word2Vec-ML_News_Upvote_Prediction/blob/master/results/regression_pt_results.png)

Linear Regression outperforms the other algorithms in both cases. On the other hand, the pre-trained model does not enhance the performances of algorithms except Decision Tree.

&nbsp;

## Possible Improvements
1- In the feature engineering step of pre-trained models, the average of the row calculated only the words that exist in pre-trained model's corpora. In order to prevent losing information at this stage, a hybrid approach can be used.

2- Other techniques to represent words as vectors such as BOW, TF-IDF etc. can be used.

3- Hyper-parameters of ML algorithms can be tweaked by using Grid Search or Random Search.
