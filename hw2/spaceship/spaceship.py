import joblib
import pandas as pd

#for submission

#處理遺漏值
df_test=pd.read_csv("spaceship train/train.csv") #將資料丟進來
df_test.drop(['Name'], axis=1, inplace=True)#砍掉非數字之資料
df_test.isnull().sum()
df_test.isnull().sum()>200#找誰的資料缺漏超過200=true
df_test.drop(['HomePlanet','CryoSleep','ShoppingMall'],axis=1,inplace=True)#缺太多也不採計
df_test.drop(['Cabin'], axis=1, inplace=True)#砍掉非純數字之資料

#填Destination空值
df_test.isnull().sum()
df_test['Destination'].value_counts().idxmax()
df_test['Destination'].fillna(df_test['Destination'].value_counts().idxmax(),inplace=True)
df_test['Destination'].value_counts()
df_test.isnull().sum()#呈現填完後結果

#填VIP空值
df_test.isnull().sum()
df_test['VIP'].value_counts().idxmax()
df_test['VIP'].fillna(df_test['VIP'].value_counts().idxmax(),inplace=True)
df_test['VIP'].value_counts()
df_test.isnull().sum()#呈現填完後結果

df_test.groupby('VIP')['Age'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['Age'].fillna(df_test.groupby('VIP')['Age'].transform('median'), inplace=True)
df_test.isnull().sum()



#填RoomService空值(同age手法)
df_test.groupby('VIP')['RoomService'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['RoomService'].fillna(df_test.groupby('VIP')['RoomService'].transform('median'), inplace=True)
df_test.isnull().sum()

#填FoodCourt空值(同age手法)
df_test.groupby('VIP')['FoodCourt'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['FoodCourt'].fillna(df_test.groupby('VIP')['FoodCourt'].transform('median'), inplace=True)
df_test.isnull().sum()

#填Spa空值(同age手法)
df_test.groupby('VIP')['Spa'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['Spa'].fillna(df_test.groupby('VIP')['Spa'].transform('median'), inplace=True)
df_test.isnull().sum()

#填VRDeck空值(同age手法)
df_test.groupby('VIP')['VRDeck'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['VRDeck'].fillna(df_test.groupby('VIP')['VRDeck'].transform('median'), inplace=True)
df_test.isnull().sum()


#將Destination  VIP進行轉換(非數字)
df_test=pd.get_dummies(data=df_test, dtype=int, columns=['Destination','VIP'])
#要轉換的資料為何 型態 哪些欄位
df_test.head()
df_test.drop('VIP_False',axis=1,inplace=True)#相對的資料只留一個
df_test.head()


#開始機器學習 資料切割
df_test.corr()#所有欄位兩兩配對看結果 畫圖 關聯性 主要是看生存
X=df_test.drop(['Transported'],axis=1)#跟Transported高度相關丟掉    #,'Destination_55 Cancri e'
y=df_test['Transported']#y是目標值，所以X不能有Y
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=67)#XY個切一刀 37分

#開始選擇模型 須為2選1
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=200)#參數為版本問題 最大取樣次數
lr.fit(X,y)#把要訓練的丟進去
predictions=lr.predict(X)

#計算結果
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y,predictions)
recall_score(y,predictions)
precision_score(y,predictions)
pd.DataFrame(confusion_matrix(y,predictions),columns=['Predictnot Survived', 'PredictSurvived'],index=['Truenot Survived','TrueSurvived'])
#可計算準確度

#Model Export輸出模型
import joblib
joblib.dump(lr,'spaceship-LR-20231115.pkl',compress=3)#compress決定壓縮比，數字越大檔案壓縮越多




