#Model Using把訓練資料丟進來
import joblib
model_pretrained=joblib.load('spaceship-LR-20231115.pkl')
import pandas as pd

#for submission
df_test=pd.read_csv("spaceship train/test.csv") #將資料丟進來
df_test.drop(['Name'], axis=1, inplace=True)#砍掉非數字之資料
df_test.isnull().sum()
df_test.drop(['Cabin','HomePlanet','CryoSleep','ShoppingMall'], axis=1, inplace=True)#缺太多也不採計
df_test.isnull().sum()

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

#填HomePlanet空值
#df_test.isnull().sum()
#df_test['HomePlanet'].value_counts().idxmax()
#df_test['HomePlanet'].fillna(df_test['HomePlanet'].value_counts().idxmax(),inplace=True)
#df_test['HomePlanet'].value_counts()
#df_test.isnull().sum()#呈現填完後結果

#填CryoSleep空值
#df_test.isnull().sum()
#df_test['CryoSleep'].value_counts().idxmax()
#df_test['CryoSleep'].fillna(df_test['CryoSleep'].value_counts().idxmax(),inplace=True)
#df_test['CryoSleep'].value_counts()
#df_test.isnull().sum()#呈現填完後結果

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

#填FoodCourt空值(同age手法)
df_test.groupby('VIP')['Spa'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['Spa'].fillna(df_test.groupby('VIP')['Spa'].transform('median'), inplace=True)
df_test.isnull().sum()

#填VRDeck空值(同age手法)
df_test.groupby('VIP')['VRDeck'].median()#.plot(kind='bar')#看VIP之中位數及畫出來
#分群後，缺的以各自中位數填補
df_test['VRDeck'].fillna(df_test.groupby('VIP')['VRDeck'].transform('median'), inplace=True)
df_test.isnull().sum()

df_test.info()

#改變資料型態
df_test=pd.get_dummies(data=df_test, dtype=int, columns=['Destination','VIP'])
df_test.drop('VIP_False', axis=1, inplace=True)#一體兩面
predictions2 = model_pretrained.predict(df_test)


#Prepare submit file
forSubmissionDF = pd.DataFrame(columns=['PassengerId','Transported'])
forSubmissionDF['PassengerId'] = range(0,4277)
forSubmissionDF['Transported'] = predictions2
forSubmissionDF

forSubmissionDF.to_csv('spaceship_for_submission_231115.csv', index=False)
