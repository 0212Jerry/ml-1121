#Model Using把上課的訓練資料丟進來
import joblib
model_pretrained=joblib.load('Titanic-LR-20231101.pkl')
import pandas as pd

#for submission
df_test=pd.read_csv("train/test.csv") #將資料丟進來
df_test.drop(['Name','Ticket'], axis=1, inplace=True)#砍掉非數字之資料
df_test.drop('Cabin', axis=1, inplace=True)#缺太多也不採計
df_test.isnull().sum()

#填剩餘的空值
df_test['Age'].fillna(df_test.groupby('Sex')['Age'].transform('median'), inplace=True)
df_test['Fare'].fillna(df_test['Fare'].value_counts().idxmax(), inplace=True)
df_test.info()#確認都填完了

#男女表示成是否為男(改變資料型態)
df_test=pd.get_dummies(data=df_test, dtype=int, columns=['Sex','Embarked'])
df_test.drop('Sex_female', axis=1, inplace=True)#男女一體兩面，丟女生
df_test.drop('Pclass', axis=1, inplace=True)
predictions2 = model_pretrained.predict(df_test)

#Prepare submit file
forSubmissionDF = pd.DataFrame(columns=['PassengerID','Survived'])
forSubmissionDF['PassengerID'] = range(892,1310)
forSubmissionDF['Survived'] = predictions2
forSubmissionDF

forSubmissionDF.to_csv('titanic_for_submission_231101.csv', index=False)

