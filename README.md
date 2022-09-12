# KK-Box-Music-Recommendation-System
Introduction to Music Recommendation System

We use the KKBOX dataset to build a music recommendation system in this project. 
This music recommendation app project will walk you through some Machine learning techniques that one can apply to recommend songs to users based 
on their listening patterns. To predict the chance of a user listening to a piece of music repetitively after the first observable listening event within 
a particular time. 

# Music Recommendation Dataset
The dataset used is from Asia’s leading music streaming service, KKBOX. It holds the world’s most comprehensive Asia-Pop music library with over 30 million tracks. 
The training data set contains the first observable listening event for each unique user-song pair within a specific time duration. Metadata of each user and song pair
is also provided. There are three datasets available.

train.csv: It contains data for different users with attributes such as msno, user_id, song_id, source_system_tab, etc. 
There are about 7.3 million entries available with 30755 unique user ids.

songs.csv: It contains the data related to songs with attributes such as song_id, song_length, genre_ids, artist_name, etc. 
The dataset contains about 2.2 million unique song ids.

members.csv: The data is related to users' information over 34403 different users.


Tech Stack for the Music Recommendation Project
Language: Python

Libraries: sklearn, xgboost, pandas, NumPy

An Outline of the Music Recommendation System Source Code
Exploratory data analysis (EDA)

#Data visualization

#Inference about features

#Feature engineering 

#Data cleaning (outlier/missing values/categorical)

#Outlier detection and treatment

#Imputing missing values

#Replacing by mode

#Removing null values

#Making a new label as missing

#Converting labeled or string values by numerical values

#Model building on training data

#Logistic regression

#Decision Tree

#Random Forest

#XGBoost

#Model validation

#Roc_Auc

#Feature importance and conclusion

#Learning Takeaways from Music Recommendation System using Machine Learning Project
This project can easily make it to the list of top machine learning projects for beginners because of the simple tools and techniques used 
to implement a music recommendation system in Python. Here are details of the machine learning tools and techniques used in this project.

#Exploratory Data Analysis
The dataset for the music recommender system project has about 3 million rows, and such large-scale data can be easily analyzed using Pandas dataframes in Python. The analysis involves understanding app-user behavior and, more precisely, what makes a user listen to songs again and again. We will achieve this by plotting insightful plots using Python libraries, matplotlib, and seaborn. For this project though, we will be using the first 10,000 rows only.

#Data Cleaning
The music recommendation system dataset has a lot of missing values that must be treated mathematically before serving the values as an input to a machine learning model. This project will help you learn three powerful techniques to handle null values in the data. You will also learn how to handle non-numerical data and treat outliers in the dataset. Additionally, you will learn how to perform feature engineering over the dataset and prepare it to apply machine learning algorithms.

#Machine Learning Algorithms
The task in this music recommendation system using python project simplifies predicting the value of a target variable which takes value '1' if the user 
listened to a particular song and '0' if they didn’t. It helps design the recommendation system as songs rows that correspond to the target value = ‘1’ are 
likely to be heard by the user and should be recommended more often. As the prediction problem falls under the umbrella of binary classification problems, you 
will explore classification machine learning algorithms: decision tree, logistic regression, XGBoost, and Random forests. After their implementation, you will 
learn how to compare the performance of different algorithms using statistical scores.
