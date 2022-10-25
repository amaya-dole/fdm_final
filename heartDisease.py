#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("HeartDisease.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')

# print(X,y)
#split set traing set 30% testingset 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
rand_forest = RandomForestClassifier()


#import pickel to train
rand_forest.fit(X_train, y_train)

#not in video
#inputt=[int(x) for x in "45 32 60".split(' ')]
#final=[np.array(inputt)]

#dump model in write mode-wb
pickle.dump(rand_forest,open('heartDiseaseModel.pkl','wb'))
#readable mode-rb
model=pickle.load(open('heartDiseaseModel.pkl','rb'))


