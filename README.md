# EXNO:4-DS
~~~
Developed ny : W Allen Johnston Ozario
Reg. No : 212224110004
~~~
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
~~~
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income.csv",na_values=[ " ?"])
data
~~~
<img width="1406" height="837" alt="image" src="https://github.com/user-attachments/assets/f7f449ba-ca07-4ab1-bc92-2ec294be5e5c" />

~~~
data.isnull().sum()
~~~
<img width="353" height="548" alt="image" src="https://github.com/user-attachments/assets/0751479a-4c0d-4ac3-a6bd-9210c0df4594" />

~~~
missing=data[data.isnull().any(axis=1)]
missing
~~~
<img width="1424" height="666" alt="image" src="https://github.com/user-attachments/assets/780de058-d61b-4107-a41e-ca11a099c8b2" />

~~~
data2=data.dropna(axis=0)
data2
~~~
<img width="1454" height="427" alt="image" src="https://github.com/user-attachments/assets/d7dfc887-0294-4806-baca-da21de78d7dc" />

~~~
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
~~~
<img width="1134" height="325" alt="image" src="https://github.com/user-attachments/assets/77172446-1a76-486e-b1ec-be6c065dd301" />

~~~
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
~~~
<img width="461" height="427" alt="image" src="https://github.com/user-attachments/assets/4c6d56c8-c0fe-4dac-897c-37ded48a96aa" />

~~~
data2
~~~
<img width="1333" height="423" alt="image" src="https://github.com/user-attachments/assets/5f022cc5-a983-42c3-ac76-f8dc5148efb0" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1706" height="279" alt="image" src="https://github.com/user-attachments/assets/9dbcd509-e049-42cc-99a0-2d7660c3182e" />

~~~
columns_list=list(new_data.columns)
print(columns_list)
~~~
o/p
```
['age', 'capitalgain', 'capitalloss', 'hoursperweek', 'SalStat', 'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'gender_ Male', 'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia']
```

~~~
features=list(set(columns_list)-set(['SalStat']))
print(features)
~~~
o/p
~~~
['nativecountry_ France', 'nativecountry_ Iran', 'nativecountry_ Jamaica', 'relationship_ Wife', 'nativecountry_ Taiwan', 'JobType_ State-gov', 'hoursperweek', 'EdType_ Some-college', 'relationship_ Unmarried', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Poland', 'nativecountry_ Honduras', 'nativecountry_ Ireland', 'JobType_ Self-emp-not-inc', 'occupation_ Protective-serv', 'age', 'occupation_ Transport-moving', 'EdType_ 9th', 'occupation_ Exec-managerial', 'nativecountry_ Japan', 'nativecountry_ Hong', 'occupation_ Machine-op-inspct', 'nativecountry_ Thailand', 'race_ Other', 'EdType_ Masters', 'JobType_ Self-emp-inc', 'occupation_ Farming-fishing', 'EdType_ Assoc-voc', 'nativecountry_ Scotland', 'occupation_ Handlers-cleaners', 'maritalstatus_ Widowed', 'nativecountry_ China', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ England', 'maritalstatus_ Never-married', 'EdType_ 1st-4th', 'EdType_ 11th', 'nativecountry_ Cuba', 'JobType_ Without-pay', 'nativecountry_ South', 'race_ White', 'nativecountry_ Columbia', 'EdType_ Doctorate', 'occupation_ Prof-specialty', 'EdType_ Preschool', 'occupation_ Sales', 'JobType_ Local-gov', 'capitalloss', 'nativecountry_ Puerto-Rico', 'gender_ Male', 'relationship_ Not-in-family', 'EdType_ Prof-school', 'EdType_ HS-grad', 'nativecountry_ El-Salvador', 'maritalstatus_ Married-civ-spouse', 'nativecountry_ Yugoslavia', 'nativecountry_ Guatemala', 'EdType_ Bachelors', 'race_ Black', 'occupation_ Armed-Forces', 'capitalgain', 'nativecountry_ Canada', 'nativecountry_ Nicaragua', 'occupation_ Craft-repair', 'maritalstatus_ Married-spouse-absent', 'nativecountry_ Italy', 'maritalstatus_ Married-AF-spouse', 'nativecountry_ Portugal', 'occupation_ Tech-support', 'nativecountry_ Laos', 'nativecountry_ Hungary', 'nativecountry_ Vietnam', 'nativecountry_ Ecuador', 'nativecountry_ United-States', 'EdType_ Assoc-acdm', 'nativecountry_ Germany', 'nativecountry_ Greece', 'race_ Asian-Pac-Islander', 'nativecountry_ Haiti', 'maritalstatus_ Separated', 'EdType_ 7th-8th', 'nativecountry_ Mexico', 'EdType_ 12th', 'nativecountry_ Trinadad&Tobago', 'occupation_ Other-service', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'JobType_ Private', 'occupation_ Priv-house-serv', 'relationship_ Own-child', 'relationship_ Other-relative', 'nativecountry_ India', 'nativecountry_ Dominican-Republic', 'EdType_ 5th-6th']
~~~

~~~
y=new_data['SalStat'].values
print(y)
~~~
<img width="203" height="36" alt="image" src="https://github.com/user-attachments/assets/4d3aa14c-619e-49ae-91dd-881ad13d7bfc" />

~~~
x=new_data[features].values
print(x)
~~~
<img width="431" height="143" alt="image" src="https://github.com/user-attachments/assets/36bbe048-726d-425a-9f16-7cdbe6b488ae" />

~~~
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
~~~
<img width="381" height="106" alt="image" src="https://github.com/user-attachments/assets/d631f036-552c-4363-b202-5b9bf547848a" />

~~~
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
~~~
<img width="223" height="63" alt="image" src="https://github.com/user-attachments/assets/32c3f959-2a13-42e5-a2d6-5cd47fb6532a" />

~~~
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
~~~
<img width="252" height="40" alt="image" src="https://github.com/user-attachments/assets/8b9f1460-e4f5-48a6-b175-fdd81cd8597b" />

~~~
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
~~~
<img width="387" height="42" alt="image" src="https://github.com/user-attachments/assets/38a12260-a739-4e08-8036-a992c7bafc20" />

~~~
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
~~~
o/p
~~~
Selected Features:
Index(['Feature3'], dtype='object')
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
~~~

~~~
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
~~~
<img width="689" height="256" alt="image" src="https://github.com/user-attachments/assets/eb832deb-4dbd-478a-82bc-ae5a93acfb44" />

~~~
tips.time.unique()
~~~
<img width="537" height="64" alt="image" src="https://github.com/user-attachments/assets/21734b24-8397-44bf-895f-f35a31dfb2f2" />

~~~
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
~~~
<img width="313" height="112" alt="image" src="https://github.com/user-attachments/assets/7f17a4d1-ac0b-42ee-b02f-06803106f1a6" />

~~~
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
~~~
<img width="468" height="70" alt="image" src="https://github.com/user-attachments/assets/7815b8d5-6ee0-4df3-9f35-684f62ab2b76" />



# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
