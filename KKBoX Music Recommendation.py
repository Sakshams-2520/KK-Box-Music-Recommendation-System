#!/usr/bin/env python
# coding: utf-8

# ## WSDM - KKBox's Music Recommendation Challenge
# 
# In this task, you will be asked to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.
# 
# KKBOX provides a training data set which consists of information of the first observable listening event for each unique user-song pair within a specific time duration. Metadata of each unique user and song pair is also provided. 
# 
# 
# Tables
# #### main.csv
# * msno: user id
# * song_id: song id
# * source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
# * source_screen_name: name of the layout a user sees.
# * source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc.
# * target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise .
# 
# 
# #### songs.csv
# * The songs. Note that data is in unicode.
# 
# * song_id
# * song_length: in ms
# * genre_ids: genre category. Some songs have multiple genres and they are separated by |
# * artist_name
# * composer
# * lyricist
# * language
# 
# #### members.csv
# user information.
# * msno
# * city
# * bd: age. Note: this column has outlier values, please use your judgement.
# * gender
# * registered_via: registration method
# * registration_init_time: format %Y%m%d
# * expiration_date: format %Y%m%d
# 

# ## Login to TrueFoundry  🎉
# 
# 1. An account with  <a href="https://projectpro.truefoundry.com/signin">TrueFoundry</a>. has been created with the same email address that you use to sign in to ProjectPro and an email has been sent to you to set your password. 
# 2. Please go to your inbox and follow the link to make sure you are logged into TrueFoundry before getting to the next cell. If you don't see the email in your inbox, please check your Spam folder. 
# 
# Note: If you are not able to signin or did not receive an email, please send an email to nikunj@truefoundry.com with the following subject- "ProjectPro User: TrueFoundry Login Issue"

# In[ ]:



get_ipython().system('pip install mlfoundry --upgrade')


# In[ ]:


import mlfoundry as mlf

TRACKING_URL = 'https://projectpro.truefoundry.com'
mlf_api = mlf.get_client(TRACKING_URL)


# In[ ]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlfoundry as mlf


# In[4]:


from zipfile import ZipFile
import urllib.request
from io import BytesIO
folder = urllib.request.urlopen("https://s3.amazonaws.com/projex.dezyre.com/Build%20a%20Music%20Recommendation%20Algorithm%20using%20KKBox's%20Dataset/materials/data.zip")
zipfile = ZipFile(BytesIO(folder.read()))
zipfile.namelist()


# In[ ]:


# loading our train_dataset
train_music = pd.read_csv("https://s3.amazonaws.com/projex.dezyre.com/music-recommendation-challenge/materials/train.csv")
train_music.rename(columns={"msno": "user_id"},inplace=True) # renaming column for better understanding
train_music.head()


# ## Exploring the dataset

# ###  Train_data

# In[ ]:


print("Length of the Dataset: ", len(train_music))


def unique_in_the_column(data):
    '''
    This function will return length of the unique values in the respective columns 
    '''
    for col in data.columns:
        print("Unqiue ", col, ":", len(data[col].unique()))


#getting unique values of the columns in train_data
unique_in_the_column(train_music)


# In[ ]:


train_music.describe(include='all')


# In[ ]:


print("Total Null values in the train_dataset:\n", train_music.isnull().sum())


# In[ ]:


plt.figure(figsize=(20,8))
plot = sns.countplot(x ='source_system_tab', data = train_music,hue="target")
plt.title('Count plot source system tabs for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * Users listen to more songs which are stored in their library
# * Songs stored in the library are the ones which user tend to listen again in a given time frame, other than that all other sources are less likely to be used

# In[ ]:


plt.figure(figsize=(30,8))
plot = sns.countplot(x ='source_type', data = train_music,hue="target")
plt.title('Count plot source_type for listening music',fontsize=10)
plt.xticks(rotation='45')

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * Local library is most visited source type for users to listen to songs
# * Local Library is the place where user visit to listen to songs again other than that other source type are less likely to be visted

# In[ ]:


plt.figure(figsize=(35,10))
plot = sns.countplot(x ='source_screen_name', data = train_music, hue="target")
plt.title('Count plot source screen name for listening music',fontsize=10)
plt.xticks(rotation='45')

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * Local playlist is most visited screen for users to listen to songs

# In[ ]:


plt.figure(figsize=(10,7))
plot = sns.barplot(x = train_music.columns, y = train_music.isnull().sum())
plt.title('Null values in each columns',fontsize=10)
plt.xticks(rotation='45')

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# In[ ]:


plt.figure(figsize=(10,7))
plot = sns.countplot(x ='target', data = train_music)
plt.title('Count plot target for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * Both targets have almost same size in the dataset

# ### On Song Data

# In[ ]:


# importing songs.csv file 
songs_data = pd.read_csv("data/songs.csv")
songs_data.head()


# In[ ]:


#getting unique values of the columns in songs_data
unique_in_the_column(songs_data)


# In[ ]:


# missing values in the data
print("Length of the song_data:",len(songs_data))
print("Total Null values in the song_dataset:\n", songs_data.isnull().sum())


# In[ ]:


# combining train data and song data to visualize language
combined_train_song_data = pd.merge(train_music, songs_data, on='song_id')


# In[ ]:


plt.figure(figsize=(25,10))
plot = sns.countplot(x ='language', data = combined_train_song_data,hue='target')
plt.title('Count plot Language for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * Song in Language 3 are the most heard songs

# ### On Members Data

# In[ ]:


# importing member.csv file 
member_data = pd.read_csv("https://s3.amazonaws.com/projex.dezyre.com/music-recommendation-challenge/materials/members.csv")
member_data.rename(columns={"msno": "user_id"},inplace=True) # renaming column for better understanding
member_data.head()


# In[ ]:


plt.figure(figsize = (8, 8))
pp = pd.value_counts(member_data.gender)
pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow=False, explode=(0.05, 0.05))
plt.axis('equal')
plt.show()


# In[ ]:


# extracting year, month and column from the string in registration_init_timr and expiration_date
member_data['registration_year'] = member_data['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
member_data['registration_expire_year'] = member_data['expiration_date'].apply(lambda x: int(str(x)[0:4]))

member_data['registration_month'] = member_data['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
member_data['expiration_month'] = member_data['expiration_date'].apply(lambda x: int(str(x)[4:6]))

member_data['registration_date'] = member_data['registration_init_time'].apply(lambda x: int(str(x)[6:]))
member_data['registration_expiration_date'] = member_data['expiration_date'].apply(lambda x: int(str(x)[6:]))


member_data.drop(['registration_init_time','expiration_date'],axis=1,inplace=True)

member_data.head()


# In[ ]:


#getting unique values of the columns in songs_data
unique_in_the_column(member_data)


# In[ ]:


# missing values in the data
print("Length of the song_data:",len(member_data))
print("Total Null values in the song_dataset:\n", member_data.isnull().sum())


# In[ ]:


# combining member data and train data for better visualization
combined_train_member_data = pd.merge(train_music, member_data, on='user_id')


# In[ ]:


plt.figure(figsize=(20,10))
plot = sns.countplot(x ='registered_via', data = combined_train_member_data,hue='target')
plt.title('Count plot registered_via for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * Most registrations were via 7 and 9 and very few with 13

# In[ ]:


plt.figure(figsize=(40,10))
plot = sns.countplot(x ='city', data = combined_train_member_data,hue='target')
plt.title('Count plot City for listening music',fontsize=30)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * People belonging to city 1 are the ones who use the app more

# In[ ]:


plt.figure(figsize=(10,8))
plot = sns.countplot(x ='gender', data = combined_train_member_data,hue='target')
plt.title('Count plot gender for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * The number of male are more in the dataset and they are more likely to listen to the same song again than females

# In[ ]:


# checking distribution of registration time 
sns.displot(combined_train_member_data, x="registration_year",bins=10).set(title='Dsitribution of Registration time')


# * More Registration can be seen from year 2012 to 2016

# In[ ]:


# checking outiers through z-score

mean_of_registration_time = np.mean(combined_train_member_data['registration_year']) 
std_of_registration_time = np.std(combined_train_member_data['registration_year'])
print("Mean of Registration Time: ",mean_of_registration_time)
print("Standard Deviation of Registartion Time: ",std_of_registration_time)

threshold = 3
outlier = [] 
for i in combined_train_member_data['registration_year']: 
    z = (i-mean_of_registration_time)/std_of_registration_time
    if z > threshold: 
        outlier.append(i) 
print('Total outlier in dataset are: ', len(outlier))


# * There are no outliers in the dataset in terms of registration date

# In[ ]:


# checking distribution of bd (age) 
sns.displot(combined_train_member_data, x="bd",bins=50).set(title='Dsitribution of Age')

# checking outiers through z-score

mean_of_age = np.mean(combined_train_member_data['bd']) 
mode_of_age = combined_train_member_data['bd'].mode()[0]
std_of_age = np.std(combined_train_member_data['bd'])
print("Mean of Age: ",mean_of_age)
print("Mode of Age: ",mode_of_age)
print("Standard Deviation of Age: ",std_of_age)

threshold = 3
outlier = [] 
for i in combined_train_member_data['bd']: 
    z = (i-mean_of_age)/std_of_age
    if z > threshold: 
        outlier.append(i) 
print('Total outlier in dataset are: ', len(outlier))
print("Maximum Age Outlier: ", max(outlier))
print("Minimum Age Outlier: ", min(outlier))


# * According to z-score there are total 6953 outliers in the dataset with respect to age column.
# * Age between 83 to 1051 are the outliers and can be removed

# In[ ]:


plt.figure(figsize=(30,8))
plot = sns.countplot(x ='registration_year', data = combined_train_member_data,hue='target')
plt.title('Count plot registration_year for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * As we can see in the above plot that no matter what registartion year is the target 1 and 0 are almost balanced

# ## Feature Engineering:

# ### Seperating genre_ids, Artist Names, Composer, Lyricist and counting them as in a single row there are more than one 

# In[ ]:


# example for genre_ids
songs_data['genre_ids'].head(10) # for eg. the 5th and 9th have 2 or more genre ids


# In[ ]:


def counting_genre_ids_artist_composer_lyricist(data):
    count = []
    for ids in data:
        try:
            if len(ids) != 0:
                count_ids = 1
                for i in ids:
                    if i == '|':
                        count_ids+=1
                count.append(count_ids)
        except TypeError:
            count.append(0)
    return count

songs_data['count_of_genre_ids'] = counting_genre_ids_artist_composer_lyricist(songs_data['genre_ids'])
songs_data['count_of_artist'] = counting_genre_ids_artist_composer_lyricist(songs_data['artist_name'])
songs_data['count_of_composer'] = counting_genre_ids_artist_composer_lyricist(songs_data['composer'])
songs_data['count_of_lyricist'] = counting_genre_ids_artist_composer_lyricist(songs_data['lyricist'])

songs_data.head()


# In[ ]:


songs_data.head(10)


# ### Removing Age 
# * between 83 to 1051 as they are outliers in the members dataset
# * Age which are equal to 0 and less than 0
# 

# In[ ]:


member_data.drop(member_data[member_data['bd'] > 82].index, inplace = True)
member_data.drop(member_data[member_data['bd'] <= 0].index, inplace = True)


# ## Handling Missing Values

# *we will handle missing values in three ways or make three models and see which performs the best:-
# * 1: We will use mode to replace values in the column
# * 2: we will remove them from the dataset
# * 3: We will add a new category as missing in the column

# In[ ]:


# merging all the data we have
merged_music_data = pd.merge(train_music, member_data, on='user_id')
merged_music_data = pd.merge(merged_music_data,songs_data, on='song_id')
merged_music_data.head()


# In[ ]:


print("Length of Merged Dataset: ",len(merged_music_data))
merged_music_data.isnull().sum()


# ### Calculating % of data missing in each column

# In[ ]:


percent_missing = merged_music_data.isnull().sum() * 100 / len(merged_music_data)
percent_missing


# * Column Gender, Lyricist have more than 40% data missing

# ### Filling Data with Mode

# In[ ]:


mode_merged_data = merged_music_data.copy()

for col in mode_merged_data.columns:
    mode_merged_data[col].fillna(mode_merged_data[col].mode()[0],inplace=True)


# In[ ]:


# replacing string varibales with numeric values
column = ['user_id','song_id','source_system_tab','source_screen_name','source_type','gender',
          'artist_name','composer','lyricist','genre_ids','language',"song_length"]

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()

def label_encoding(data):
    for col in column:
        print(col)
        data[col]= label_encoder.fit_transform(data[col])
        
    return data
    
label_encoding(mode_merged_data)


# ### 2: Removing All null values

# In[ ]:


removed_null_data = merged_music_data.copy()

removed_null_data.dropna(inplace = True)
print("Length of data after removing missing values: ",len(removed_null_data))

removed_null_data.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,7))
plot = sns.countplot(x ='target', data = removed_null_data)
plt.title('Count plot target for listening music',fontsize=10)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), 
               (p.get_x() + p.get_width() / 2.,
                p.get_height()),
               ha = 'center',va = 'center', xytext = (0, 10), textcoords = 'offset points')


# * The Target 0 gets reduced if we remove the null values completely

# In[ ]:


label_encoding(removed_null_data)


# ### 3: Making a new label as Missing 

# In[ ]:


missing_label_merged_data = merged_music_data.copy()

for col in missing_label_merged_data.columns:
    if col != 'language':
        missing_label_merged_data[col].fillna('missing',inplace=True)
    else:
        missing_label_merged_data[col].fillna(0,inplace=True)


# In[ ]:


missing_label_merged_data.isnull().sum()


# In[ ]:


missing_label_merged_data.head(20)


# In[ ]:


label_encoding(missing_label_merged_data)


# ## Implementing Models
# 
# * We will be using AUC Score for selecting the model as it it metric which is used on Kaggle.

# In[ ]:


from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, ConfusionMatrixDisplay


x_train_mode, x_test_mode, y_train_mode, y_test_mode = train_test_split(mode_merged_data.drop(['target']
                                                                                              ,axis=1),
                                                    mode_merged_data['target'],test_size=0.20,random_state=40)

x_train_removed_null, x_test_removed_null,y_train_removed_null, y_test_removed_null = train_test_split(removed_null_data.drop(['target'],
                                                                                                                              axis=1),
                 removed_null_data['target'],test_size=0.20,random_state=40)

x_train_missing_label, x_test_missing_label, y_train_missing_label, y_test_missing_label = train_test_split(missing_label_merged_data.drop(['target'],axis=1),
                 missing_label_merged_data['target'],test_size=0.20,random_state=40)


# In[ ]:


x_train_mode.head()


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# for mode
model.fit(x_train_mode,y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ",roc_auc_score(predict_mode,y_test_mode))

# for removed_data
model.fit(x_train_removed_null,y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)
try:
    print("Model Accuracy on Removed Null Data: ",roc_auc_score(predict_removed_null,y_test_removed_null))
except ValueError:
    print("Only One Class Present")


# for new_missing_label
model.fit(x_train_missing_label,y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",roc_auc_score(predict_missing_label,y_test_missing_label))


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Logistic-Regression-for-model')


# In[ ]:


train_dataset = x_train_mode.copy()
train_dataset['targets'] = y_train_mode
train_dataset['predictions'] = model.predict(x_train_mode)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_mode))


# In[ ]:


test_dataset = x_test_mode.copy()
test_dataset['targets'] = y_test_mode
test_dataset['predictions'] = model.predict(x_test_mode)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_mode))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_mode.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_mode.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_mode)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_mode, y_predict),
    "Precision": precision_score(y_test_mode, y_predict),
    "Recall": recall_score(y_test_mode, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_mode)[:,1]

fpr, tpr, _ = roc_curve(y_test_mode,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_mode, model.predict(x_test_mode))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Logistic-Regression-for-removed-data')


# In[ ]:


train_dataset = x_train_removed_null.copy()
train_dataset['targets'] = y_train_removed_null
train_dataset['predictions'] = model.predict(x_train_removed_null)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_removed_null))


# In[ ]:


test_dataset = x_test_removed_null.copy()
test_dataset['targets'] = y_test_removed_null
test_dataset['predictions'] = model.predict(x_test_removed_null)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_removed_null))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_removed_null.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_removed_null.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_removed_null)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_removed_null, y_predict),
    "Precision": precision_score(y_test_removed_null, y_predict),
    "Recall": recall_score(y_test_removed_null, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_removed_null)[:,1]

fpr, tpr, _ = roc_curve(y_test_removed_null,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_removed_null, model.predict(x_test_removed_null))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Logistic-Regression-for-new-missing-label')


# In[ ]:


train_dataset = x_train_missing_label.copy()
train_dataset['targets'] = y_train_missing_label
train_dataset['predictions'] = model.predict(x_train_missing_label)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_missing_label))


# In[ ]:


test_dataset = x_test_missing_label.copy()
test_dataset['targets'] = y_test_missing_label
test_dataset['predictions'] = model.predict(x_test_missing_label)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_missing_label))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_missing_label.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_missing_label.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_missing_label)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_missing_label, y_predict),
    "Precision": precision_score(y_test_missing_label, y_predict),
    "Recall": recall_score(y_test_missing_label, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_missing_label)[:,1]

fpr, tpr, _ = roc_curve(y_test_missing_label,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_missing_label, model.predict(x_test_missing_label))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# ### DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# for mode
model.fit(x_train_mode,y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ",roc_auc_score(predict_mode,y_test_mode))

# for removed_data
model.fit(x_train_removed_null,y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)
try:
    print("Model Accuracy on Removed Null Data: ",roc_auc_score(predict_removed_null,y_test_removed_null))
except ValueError:
    print("Only One Class Present")


# for new_missing_label
model.fit(x_train_missing_label,y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",roc_auc_score(predict_missing_label,y_test_missing_label))


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Decision-Tree-Classifier-for-model')


# In[ ]:


train_dataset = x_train_mode.copy()
train_dataset['targets'] = y_train_mode
train_dataset['predictions'] = model.predict(x_train_mode)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_mode))


# In[ ]:


test_dataset = x_test_mode.copy()
test_dataset['targets'] = y_test_mode
test_dataset['predictions'] = model.predict(x_test_mode)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_mode))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_mode.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_mode.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_mode)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_mode, y_predict),
    "Precision": precision_score(y_test_mode, y_predict),
    "Recall": recall_score(y_test_mode, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_mode)[:,1]

fpr, tpr, _ = roc_curve(y_test_mode,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_mode, model.predict(x_test_mode))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Decision-Tree-Classifier-for-removed-data')


# In[ ]:


train_dataset = x_train_removed_null.copy()
train_dataset['targets'] = y_train_removed_null
train_dataset['predictions'] = model.predict(x_train_removed_null)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_removed_null))


# In[ ]:


test_dataset = x_test_removed_null.copy()
test_dataset['targets'] = y_test_removed_null
test_dataset['predictions'] = model.predict(x_test_removed_null)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_removed_null))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_removed_null.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_removed_null.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_removed_null)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_removed_null, y_predict),
    "Precision": precision_score(y_test_removed_null, y_predict),
    "Recall": recall_score(y_test_removed_null, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_removed_null)[:,1]

fpr, tpr, _ = roc_curve(y_test_removed_null,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_removed_null, model.predict(x_test_removed_null))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Decision_Tree_Classifier_for_new_missing_label')


# In[ ]:


train_dataset = x_train_missing_label.copy()
train_dataset['targets'] = y_train_missing_label
train_dataset['predictions'] = model.predict(x_train_missing_label)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_missing_label))


# In[ ]:


test_dataset = x_test_missing_label.copy()
test_dataset['targets'] = y_test_missing_label
test_dataset['predictions'] = model.predict(x_test_missing_label)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_missing_label))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_missing_label.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_missing_label.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_missing_label)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_missing_label, y_predict),
    "Precision": precision_score(y_test_missing_label, y_predict),
    "Recall": recall_score(y_test_missing_label, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_missing_label)[:,1]

fpr, tpr, _ = roc_curve(y_test_missing_label,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_missing_label, model.predict(x_test_missing_label))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# ### RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

# for mode
rf_model.fit(x_train_mode,y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ",accuracy_score(predict_mode,y_test_mode))

# for removed_data
rf_model.fit(x_train_removed_null,y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",accuracy_score(predict_removed_null,y_test_removed_null))

# for new_missing_label
rf_model.fit(x_train_missing_label,y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",accuracy_score(predict_missing_label,y_test_missing_label))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 200)

# for mode
rf_model.fit(x_train_mode,y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ",accuracy_score(predict_mode,y_test_mode))

# for removed_data
rf_model.fit(x_train_removed_null,y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",accuracy_score(predict_removed_null,y_test_removed_null))

# for new_missing_label
rf_model.fit(x_train_missing_label,y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",accuracy_score(predict_missing_label,y_test_missing_label))


# In[ ]:


rf_model.auc


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Random-Forest-Classifier-for-mode')


# In[ ]:


train_dataset = x_train_mode.copy()
train_dataset['targets'] = y_train_mode
train_dataset['predictions'] = rf_model.predict(x_train_mode)
train_dataset['prediction_probabilities'] = list(rf_model.predict_proba(x_train_mode))


# In[ ]:


test_dataset = x_test_mode.copy()
test_dataset['targets'] = y_test_mode
test_dataset['predictions'] = rf_model.predict(x_test_mode)
test_dataset['prediction_probabilities'] = list(rf_model.predict_proba(x_test_mode))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_mode.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_mode.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = rf_model.predict(x_test_mode)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_mode, y_predict),
    "Precision": precision_score(y_test_mode, y_predict),
    "Recall": recall_score(y_test_mode, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(rf_model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(rf_model.get_params())


# In[ ]:


y_pred_proba = rf_model.predict_proba(x_test_mode)[:,1]

fpr, tpr, _ = roc_curve(y_test_mode,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_mode, rf_model.predict(x_test_mode))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Random-Forest-Classifier-for-removed-data')


# In[ ]:


train_dataset = x_train_removed_null.copy()
train_dataset['targets'] = y_train_removed_null
train_dataset['predictions'] = rf_model.predict(x_train_removed_null)
train_dataset['prediction_probabilities'] = list(rf_model.predict_proba(x_train_removed_null))


# In[ ]:


test_dataset = x_test_removed_null.copy()
test_dataset['targets'] = y_test_removed_null
test_dataset['predictions'] = rf_model.predict(x_test_removed_null)
test_dataset['prediction_probabilities'] = list(rf_model.predict_proba(x_test_removed_null))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_removed_null.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_removed_null.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = rf_model.predict(x_test_removed_null)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_removed_null, y_predict),
    "Precision": precision_score(y_test_removed_null, y_predict),
    "Recall": recall_score(y_test_removed_null, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(rf_model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(rf_model.get_params())


# In[ ]:


y_pred_proba = rf_model.predict_proba(x_test_removed_null)[:,1]

fpr, tpr, _ = roc_curve(y_test_removed_null,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_removed_null, rf_model.predict(x_test_removed_null))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='Random-Forest-Classifier-for-new-missing-label')


# In[ ]:


train_dataset = x_train_missing_label.copy()
train_dataset['targets'] = y_train_missing_label
train_dataset['predictions'] = rf_model.predict(x_train_missing_label)
train_dataset['prediction_probabilities'] = list(rf_model.predict_proba(x_train_missing_label))


# In[ ]:


test_dataset = x_test_missing_label.copy()
test_dataset['targets'] = y_test_missing_label
test_dataset['predictions'] = rf_model.predict(x_test_missing_label)
test_dataset['prediction_probabilities'] = list(rf_model.predict_proba(x_test_missing_label))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_missing_label.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_missing_label.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = rf_model.predict(x_test_missing_label)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_missing_label, y_predict),
    "Precision": precision_score(y_test_missing_label, y_predict),
    "Recall": recall_score(y_test_missing_label, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(rf_model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(rf_model.get_params())


# In[ ]:


y_pred_proba = rf_model.predict_proba(x_test_missing_label)[:,1]

fpr, tpr, _ = roc_curve(y_test_missing_label,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_missing_label, rf_model.predict(x_test_missing_label))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# ### XGBClassifier

# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(enable_categorical=True)

# for mode
model.fit(x_train_mode,y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ",accuracy_score(predict_mode,y_test_mode))

# for removed_data
model.fit(x_train_removed_null,y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",accuracy_score(predict_removed_null,y_test_removed_null))

# for new_missing_label
model.fit(x_train_missing_label,y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",accuracy_score(predict_missing_label,y_test_missing_label))


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='XGBClassifier-for-model')


# In[ ]:


train_dataset = x_train_mode.copy()
train_dataset['targets'] = y_train_mode
train_dataset['predictions'] = model.predict(x_train_mode)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_mode))


# In[ ]:


test_dataset = x_test_mode.copy()
test_dataset['targets'] = y_test_mode
test_dataset['predictions'] = model.predict(x_test_mode)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_mode))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_mode.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_mode.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_mode)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_mode, y_predict),
    "Precision": precision_score(y_test_mode, y_predict),
    "Recall": recall_score(y_test_mode, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_mode)[:,1]

fpr, tpr, _ = roc_curve(y_test_mode,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_mode, model.predict(x_test_mode))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='XGBClassifier-for-removed-data')


# In[ ]:


train_dataset = x_train_removed_null.copy()
train_dataset['targets'] = y_train_removed_null
train_dataset['predictions'] = model.predict(x_train_removed_null)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_removed_null))


# In[ ]:


test_dataset = x_test_removed_null.copy()
test_dataset['targets'] = y_test_removed_null
test_dataset['predictions'] = model.predict(x_test_removed_null)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_removed_null))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_removed_null.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_removed_null.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_removed_null)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_removed_null, y_predict),
    "Precision": precision_score(y_test_removed_null, y_predict),
    "Recall": recall_score(y_test_removed_null, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_removed_null)[:,1]

fpr, tpr, _ = roc_curve(y_test_removed_null,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_removed_null, model.predict(x_test_removed_null))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='XGBClassifier-for-new-missing-label')


# In[ ]:


train_dataset = x_train_missing_label.copy()
train_dataset['targets'] = y_train_missing_label
train_dataset['predictions'] = model.predict(x_train_missing_label)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_missing_label))


# In[ ]:


test_dataset = x_test_missing_label.copy()
test_dataset['targets'] = y_test_missing_label
test_dataset['predictions'] = model.predict(x_test_missing_label)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_missing_label))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_missing_label.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_missing_label.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_missing_label)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_missing_label, y_predict),
    "Precision": precision_score(y_test_missing_label, y_predict),
    "Recall": recall_score(y_test_missing_label, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_missing_label)[:,1]

fpr, tpr, _ = roc_curve(y_test_missing_label,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_missing_label, model.predict(x_test_missing_label))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# # XGBoost

# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=2)

# for mode
model.fit(x_train_mode, y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ", accuracy_score(predict_mode,
                                                      y_test_mode))

# for removed_data
model.fit(x_train_removed_null, y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",
      accuracy_score(predict_removed_null, y_test_removed_null))

# for new_missing_label
model.fit(x_train_missing_label, y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",
      accuracy_score(predict_missing_label, y_test_missing_label))


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=5)

# for mode
model.fit(x_train_mode, y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ", accuracy_score(predict_mode,
                                                      y_test_mode))

# for removed_data
model.fit(x_train_removed_null, y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",
      accuracy_score(predict_removed_null, y_test_removed_null)) 

# for new_missing_label
model.fit(x_train_missing_label, y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",
      accuracy_score(predict_missing_label, y_test_missing_label))


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=50)

# for mode
model.fit(x_train_mode, y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ", accuracy_score(predict_mode,
                                                      y_test_mode))

# for removed_data
model.fit(x_train_removed_null, y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",
      accuracy_score(predict_removed_null, y_test_removed_null))

# for new_missing_label
model.fit(x_train_missing_label, y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",
      accuracy_score(predict_missing_label, y_test_missing_label))


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100)

# for mode
model.fit(x_train_mode, y_train_mode)
predict_mode = model.predict(x_test_mode)

print("Model Accuracy on Mode Data: ", accuracy_score(predict_mode,
                                                      y_test_mode))

# for removed_data
model.fit(x_train_removed_null, y_train_removed_null)
predict_removed_null = model.predict(x_test_removed_null)

print("Model Accuracy on Removed Null Data: ",
      accuracy_score(predict_removed_null, y_test_removed_null))

# for new_missing_label
model.fit(x_train_missing_label, y_train_missing_label)
predict_missing_label = model.predict(x_test_missing_label)

print("Model Accuracy on New Missing label Data: ",
      accuracy_score(predict_missing_label, y_test_missing_label))


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='XGBoost-for-model')


# In[ ]:


train_dataset = x_train_mode.copy()
train_dataset['targets'] = y_train_mode
train_dataset['predictions'] = model.predict(x_train_mode)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_mode))


# In[ ]:


test_dataset = x_test_mode.copy()
test_dataset['targets'] = y_test_mode
test_dataset['predictions'] = model.predict(x_test_mode)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_mode))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_mode.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_mode.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_mode)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_mode, y_predict),
    "Precision": precision_score(y_test_mode, y_predict),
    "Recall": recall_score(y_test_mode, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_mode)[:,1]

fpr, tpr, _ = roc_curve(y_test_mode,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_mode, model.predict(x_test_mode))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='XGBoost-for-removed-data')


# In[ ]:


train_dataset = x_train_removed_null.copy()
train_dataset['targets'] = y_train_removed_null
train_dataset['predictions'] = model.predict(x_train_removed_null)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_removed_null))


# In[ ]:


test_dataset = x_test_removed_null.copy()
test_dataset['targets'] = y_test_removed_null
test_dataset['predictions'] = model.predict(x_test_removed_null)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_removed_null))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_removed_null.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_removed_null.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_removed_null)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_removed_null, y_predict),
    "Precision": precision_score(y_test_removed_null, y_predict),
    "Recall": recall_score(y_test_removed_null, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_removed_null)[:,1]

fpr, tpr, _ = roc_curve(y_test_removed_null,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_removed_null, model.predict(x_test_removed_null))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# In[ ]:


mlf_run = mlf_api.create_run(project_name='music-recommendation-system', run_name='XGBoost-for-new-missing-label')


# In[ ]:


train_dataset = x_train_missing_label.copy()
train_dataset['targets'] = y_train_missing_label
train_dataset['predictions'] = model.predict(x_train_missing_label)
train_dataset['prediction_probabilities'] = list(model.predict_proba(x_train_missing_label))


# In[ ]:


test_dataset = x_test_missing_label.copy()
test_dataset['targets'] = y_test_missing_label
test_dataset['predictions'] = model.predict(x_test_missing_label)
test_dataset['prediction_probabilities'] = list(model.predict_proba(x_test_missing_label))


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'train_dataset',
    features = train_dataset[list(x_train_missing_label.columns)],
    predictions = train_dataset['predictions'],
    actuals = train_dataset['targets'],
    only_stats = False,
)


# In[ ]:


mlf_run.log_dataset(
    dataset_name = 'test_dataset',
    features = test_dataset[list(x_test_missing_label.columns)],
    predictions = test_dataset['predictions'],
    actuals = test_dataset['targets'],
    only_stats = False,
)


# In[ ]:


y_predict = model.predict(x_test_missing_label)

metrics_dict = {
    "Accuracy": accuracy_score(y_test_missing_label, y_predict),
    "Precision": precision_score(y_test_missing_label, y_predict),
    "Recall": recall_score(y_test_missing_label, y_predict),
}

mlf_run.log_metrics(metrics_dict)


# In[ ]:


mlf_run.log_model(model, framework=mlf.ModelFramework.SKLEARN)
mlf_run.log_params(model.get_params())


# In[ ]:


y_pred_proba = model.predict_proba(x_test_missing_label)[:,1]

fpr, tpr, _ = roc_curve(y_test_missing_label,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
mlf_run.log_plots({"roc-curve": plt}, step=1)
plt.show()

mat = confusion_matrix(y_test_missing_label, model.predict(x_test_missing_label))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
mlf_run.log_plots({"confusion-matrix": plt}, step=1)

mlf_run.end()


# ## Feature Importance

# In[ ]:


x_train_missing_label.columns


# In[ ]:


feature_importance = pd.DataFrame({'Features':x_train_missing_label.columns,
                                  'Importance':rf_model.feature_importances_})

feature_importance


# In[ ]:


plt.figure(figsize=(15,6))

sns.barplot(x=feature_importance['Features'], y=feature_importance['Importance'],
            data = feature_importance,order=feature_importance.sort_values('Importance').Features).set(title='Feature Importance')

plt.xticks(rotation=90)
plt.show()


# ## Conclusion:
# * Looking at the feature importance the user who is listening the song has the highest weightage
# * It's possible that on Registration day and Service Expiration day people use the app more and thus listen to songs on repeat.
# * Age (bd) is always a factor as the dataset has a mean age of 17 years and youngsters do listen to more music
# * Composer and Artist are also important features.

# In[ ]:




