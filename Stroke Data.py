#!/usr/bin/env python
# coding: utf-8

# Research Question: Can 9 classifiers be built with 75% or more accuracy with the given hospital dataset?
# 
# Data: The dataset for this analysis is publicly available information provided by Data.gov. The original data set contained 5110 rows and 12 columns.
# 
# The data set includes the following variables: "id", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke". 
# 
# There is no information that would make the hospitals associated with this analysis identifiable. 
# 
# Limitations: The dataset is limited to the patient population included in the dataset. 
# There are no delimitations to this study. 

# In[1]:


#imports
import numpy as np 
import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rc("font", size=14)
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import scipy

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, precision_recall_curve, auc,plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from colorama import Fore, Back, Style


# In[2]:


#package/library versions
print("pandas version " + pd.__version__)
print("numpy version " + np.__version__)
print("scipy version " + scipy.__version__)
print("matplotlib version " + matplotlib.__version__)
print("seaborn version " + sns.__version__)


# In[3]:


#ignore future warnings
import warnings 
warnings.filterwarnings('ignore') 


# In[4]:


#load data into pandas Dataframe
df = pd.read_csv("stroke_dataset.csv")


# In[5]:


#view top of data
df.head()


# In[6]:


df.shape


# In[7]:


#check for duplicates
df.duplicated()


# In[8]:


#check for null values
print(df.isnull().sum())


# In[9]:


#impute null 'bmi' values with mean 
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())


# In[10]:


#confirm imputation worked
print(df.isnull().sum())


# In[11]:


#check for outliers in age and bmi
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 6) , squeeze=True)

sns.boxplot(data=df,y=df['age'],palette='tab10' , ax=axes[0])
sns.boxplot(data=df,y=df['bmi'],palette='tab10' , ax=axes[1])

plt.show


# In[12]:


#address outliers in 'bmi'

#display rows with 'bmi' > 70
display(df[df['bmi'] > 70])

#drop rows  greater than 70
df.drop(df.index[df['bmi'] > 70], inplace=True)

#reset index of dataframe
df = df.reset_index(drop = True)


# In[13]:


#viewing outliers after addressing outliers in 'bmi'
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 6) , squeeze=True)

sns.boxplot(data=df,y=df['age'],palette='tab10' , ax=axes[0])
sns.boxplot(data=df,y=df['bmi'],palette='tab10' , ax=axes[1])

plt.show


# In[14]:


#delete id column as it's not necessary for analysis
df = df.drop("id",axis=1)


# In[15]:


#checking dataset balance
sns.countplot(df['stroke'], label="Count")
plt.show()


# Result: data is not balanced

# Data Visualization

# In[16]:


#boxplot
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 6) , squeeze=True)

sns.boxplot(data=df,y=df['age'],x=df['stroke'],palette='tab10' , ax=axes[0])
sns.boxplot(data=df,y=df['bmi'],x=df['stroke'],palette='tab10' , ax=axes[1])
sns.boxplot(data=df,y=df['avg_glucose_level'],x=df['stroke'],palette='tab10' , ax=axes[2])

plt.show


# In[17]:


#histograms
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (18, 18))
fig.delaxes( ax=axes[2,1])
fig.delaxes( ax=axes[2,2])
sns.countplot(x="gender", hue='stroke', palette='tab10', data=df , ax=axes[0,0])
sns.countplot(x="hypertension", hue='stroke', palette="tab10", data=df , ax=axes[0,1])
sns.countplot(x="heart_disease", hue='stroke', palette="tab10", data=df , ax=axes[0,2])

sns.countplot(x="ever_married", hue='stroke', palette="tab10", data=df , ax=axes[1,0])
sns.countplot(x="work_type", hue='stroke', palette="tab10", data=df , ax=axes[1,1])
sns.countplot(x="Residence_type", hue='stroke', palette="tab10", data=df , ax=axes[1,2])

sns.countplot(x="smoking_status", hue='stroke', palette="tab10", data=df , ax=axes[2,0])

plt.show()


# In[18]:


#scatterplots
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 6) , squeeze=True)

sns.scatterplot(data=df, x=df['stroke'], y=df['age'],palette='tab10', ax=axes[0])
sns.scatterplot(data=df, x=df['stroke'], y=df['avg_glucose_level'],palette='tab10', ax=axes[1])
sns.scatterplot(data=df, x=df['stroke'], y=df['bmi'],palette='tab10',ax=axes[2] )

plt.show


# Model building

# In[19]:


#categorical columns in the training data
object_cols = [col for col in df.columns if df[col].dtype == "object"]

print('Categorical columns that will be ordinal encoded:', object_cols)


# In[20]:


#get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: df[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))

#print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# In[21]:


ordinal_encoder = OrdinalEncoder()
df[object_cols] = ordinal_encoder.fit_transform(df[object_cols]) 


# In[22]:


#oversampling the data

smote = SMOTE()
test_df  = df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi','stroke']].sample(int(df.shape[0]*0.2),random_state=42)
train_df = df.drop(index=test_df.index)

X_test, y_test   = test_df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']], test_df['stroke']
X_train, y_train = train_df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']], train_df['stroke']


X_train, y_train = smote.fit_resample(X_train, y_train)
upsampled_df = X_train.assign(Stroke = y_train)

X_test, y_test = smote.fit_resample(X_test, y_test)
up_test_df = X_test.assign(Stroke = y_test)


# In[23]:


fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(15, 5))

df.stroke.value_counts().plot(kind='bar', color='tab:red', title='Stroke - Before Upsampling', ax=axes[0])
upsampled_df.Stroke.value_counts().plot(kind='bar', color='tab:green', title='Stroke(Train set) - After Upsampling', ax=axes[1])
up_test_df.Stroke.value_counts().plot(kind='bar', color='tab:blue', title='Stroke (Test set) - After Upsampling', ax=axes[2]);


# In[24]:


#performance comparison visualization

def score_vis(score):
    
    names = ['Naive Bayes' 'SVM', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'AdaBoost', 'XGBoost', 'CatBoost', 'KNN']

    plt.rcParams['figure.figsize']=20,8
    ax = sns.barplot(x=names, y=score, palette = "plasma", saturation =2.0)
    plt.xlabel('Model', fontsize = 20 )
    plt.ylabel('Accuracy(%)', fontsize = 20)
    plt.title('Model Comparison - Test set', fontsize = 20)
    plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
    plt.yticks(fontsize = 12)
    for i in ax.patches:
        width, height = i.get_width(), i.get_height()
        x, y = i.get_xy() 
        ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
    plt.show()


# In[25]:


def trainer(X_train, y_train, X_test, y_test):
    
    models= [[' Naive Bayes ', GaussianNB()],
             [' SVM ',SVC()],
             [' Decision Tree ', DecisionTreeClassifier()],
             [' Random Forest ', RandomForestClassifier()],
             [' Logistic Regression ', LogisticRegression(max_iter=200)],
             [' AdaBoost ', AdaBoostClassifier()],
             [' XGBoost ', XGBClassifier()],
             [' CatBoost ', CatBoostClassifier(logging_level='Silent')],
             [' KNN ', KNeighborsClassifier()]]

    scores = []
    
    for model_name, model in models:

        model = model
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cm_model = confusion_matrix(y_test, pred)
        scores.append(accuracy_score(y_test, model.predict(X_test)))

        print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + model_name)
        print(Back.RESET)
        print(cm_model)
        print('\n' + Fore.BLUE + 'Training Acc.  : ' + Fore.GREEN + str(round(accuracy_score(y_train, model.predict(X_train)) * 100, 2)) + '%' )
        print(Fore.BLUE + 'Validation Acc.: ' + Fore.RED + str(round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)) + '%\n' )
        print(Fore.CYAN + classification_report(y_test, pred)) 
        print('\n' + Fore.BLACK + Back.WHITE + '⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜⁜\n')
    
        
    return scores


# In[26]:


#results
scores = trainer(X_train, y_train, X_test, y_test)

