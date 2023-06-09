import pandas as pd
from sklearn.model_selection import train_test_split

test=pd.read_csv("https://raw.githubusercontent.com/A-Anastasia/AMO2/main/test.csv")
train=pd.read_csv("https://raw.githubusercontent.com/A-Anastasia/AMO2/main/train.csv")

df=pd.read_csv("https://raw.githubusercontent.com/A-Anastasia/AMO2/main/train.csv",index_col='id')

"""Пресутствуют ли нулевые значения"""

df.isna().sum()

x=df.iloc[:,0:-1]
y=df.iloc[:,-1]


"""Используйте метод smote, а SMOTE — это специальная библиотека, которая поможет нам сбалансировать набор данных."""

from imblearn.over_sampling import SMOTE
smt=SMOTE()
trainx,trainy=smt.fit_resample(x,y)

"""Используйте стандартный масштабатор для стандартизации данных"""

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X=scaler.fit_transform(trainx)

"""Разделяем выборки"""

x_train,x_test,y_train,y_test=train_test_split(X,trainy,test_size=0.20,random_state=42)
pd.DataFrame(x_train).to_csv("/home/asha/AMO2/x_train.csv")
pd.DataFrame(x_test).to_csv("/home/asha/AMO2/x_test.csv")
pd.DataFrame(y_train).to_csv("/home/asha/AMO2/y_train.csv")
pd.DataFrame(y_test).to_csv("/home/asha/AMO2/y_test.csv")
