#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
df=pd.read_csv("https://raw.githubusercontent.com/ryanchung403/dataset/main/train_data_titanic.csv")
# or df=pd.read_csv("data/train_data_titanic.csv")
df.head()
df.info()#Age有缺資料(714非空)

#Remove the columns model will not use 
df.drop(['Name','Ticket'],axis=1,inplace=True)#丟掉較複雜且不實用的資料(欄位名稱)。axis=1為列=1為column。inplace=True表示真的要改變df中的資料(下方jupyter資料size會變)
#df.head()

#觀察資料。畫圖觀察兩兩關係(存活)
sns.pairplot(df[['Survived','Fare']],dropna=True)#dropna=True表示不納入有缺漏的資料
sns.pairplot(df[['Survived','Pclass']],dropna=True)#非數值化的資料pairplot畫不出來
df.groupby('Survived').mean(numeric_only=True)
#groupby以死活分組，然後計算各組之平均，但有些資料沒辦法平均(非數值)，所以需要numeric_only=True 只算有數值的(python版本問題)
#存活者平均年齡稍低一些!票價平均較高一些!
df['SibSp'].value_counts()#value_counts看到資料出現的頻率(看sibsp出現的數字次數)
df['Parch'].value_counts()
df['Sex'].value_counts()

#Handle missing values處理遺漏值
df.isnull().sum()#偵測有啥遺漏值，空的就是true。if空值不多，可以填補
df.isnull().sum()>(len(df)/2)#找誰的資料缺漏超過總資料891的一半=true
#Cabin has too many missing values
df.drop('Cabin',axis=1,inplace=True)#砍掉cabin
#df['Age'].isnull().value_counts()
#Age is also have some missing values
df.groupby('Sex')['Age'].median()#.plot(kind='bar')   #看性別之中位數及畫出來
#分群後，缺的以各自中位數填補
df['Age'].fillna(df.groupby('Sex')['Age'].transform('median'), inplace=True)

#填embark空值
df.isnull().sum()
df['Embarked'].value_counts().idxmax()
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df['Embarked'].value_counts()
df.isnull().sum()#呈現填完後結果

#將Sex,Embarked進行轉換(非數字)
df=pd.get_dummies(data=df, dtype=int, columns=['Sex','Embarked'])
#要轉換的資料為何 型態 哪些欄位
df.head()
df.drop('Sex_female',axis=1,inplace=True)#男女是相對的，只留是否為男生，砍掉女生
df.head()

#開始機器學習 資料切割
df.corr()#所有欄位兩兩配對看結果 畫圖 關聯性 主要是看生存
X=df.drop(['Survived','Pclass'],axis=1)#Pclass跟Survive高度相關  丟掉
y=df['Survived']#y是目標值，所以X不能有Y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=67)#XY個切一刀 37分

#開始選擇模型 須為2選1
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=200)#參數為版本問題 最大取樣次數
lr.fit(X_train,y_train)#把要訓練的丟進去
predictions=lr.predict(X_test)

#計算結果
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)
pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predictnot Survived', 'PredictSurvived'],index=['Truenot Survived','TrueSurvived'])
#可計算準確度

#Model Export輸出模型
import joblib
joblib.dump(lr,'Titanic-LR-20231101.pkl',compress=3)#compress決定壓縮比，數字越大檔案壓縮越多

