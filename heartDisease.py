import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("HeartDisease.csv")
x = data.drop('target',axis=1)
y = data['target']

# print(X,y)
#split set traing set 30% testingset 70%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
rand_forest = RandomForestClassifier()


#import pickel to train
rand_forest.fit(X_train, y_train)


#dump model in write mode-wb
pickle.dump(rand_forest,open('heartPrediction.pkl','wb'))
#readable mode-rb
model=pickle.load(open('heartPrediction.pkl','rb'))
