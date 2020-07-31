# import necessary modules
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# load the data set
# data = pd.read_csv('MesoData/shakir_data.csv')
# data = pd.read_csv('MesoData/RandomOverSampling.csv')
data = pd.read_csv('data/SMOTEAZURE.csv')
# data = pd.read_csv('MesoData/SMOTE_Python.csv')


# Undersampling
# data = pd.read_csv('MesoData/InstanceHardnessThreshold.csv')
# data = pd.read_csv('MesoData/RandomUnderSampler.csv')
# data = pd.read_csv('data/dataset_1.csv')

# print info about columns in the dataframe
# print(data.info())


# normalise the amount column
# data['platelet count (PLT)'] = StandardScaler().fit_transform(np.array(data['platelet count (PLT)']).reshape(-1, 1))
# data['keep side'] = StandardScaler().fit_transform(np.array(data['keep side']).reshape(-1, 1))
# data['class_of_diagnosis'] = StandardScaler().fit_transform(np.array(data['class_of_diagnosis']).reshape(-1, 1))

# drop Time and Amount columns as they are not relevant for prediction purpose
data = data.drop(['diagnosis method'], axis = 1)


# Get Top 10 Releif F Features
# data = data.drop(['gender'], axis = 1)
# data = data.drop(['ache on chest'], axis = 1)
# data = data.drop(['city'], axis = 1)
# data = data.drop(['weakness'], axis = 1)
# data = data.drop(['pleural effusion'], axis = 1)
# data = data.drop(['habit of cigarette'], axis = 1)
# data = data.drop(['cytology'], axis = 1)
# data = data.drop(['blood lactic dehydrogenise (LDH)'], axis = 1)
# data = data.drop(['performance status'], axis = 1)
#
# data = data.drop(['pleural protein'], axis = 1)
# data = data.drop(['keep side'], axis = 1)
# data = data.drop(['C-reactive protein (CRP)'], axis = 1)
# data = data.drop(['pleural thickness on tomography'], axis = 1)
# data = data.drop(['white blood'], axis = 1)
# data = data.drop(['pleural albumin'], axis = 1)
# data = data.drop(['sedimentation'], axis = 1)
# data = data.drop(['asbestos exposure'], axis = 1)
# data = data.drop(['pleural glucose'], axis = 1)
# data = data.drop(['pleural level of acidity (pH)'], axis = 1)
# data = data.drop(['duration of asbestos exposure'], axis = 1)
# data = data.drop(['age'], axis = 1)
# data = data.drop(['cell count (WBC)'], axis = 1)
# data = data.drop(['hemoglobin (HGB)'], axis = 1)
# data = data.drop(['type of MM'], axis = 1)
# data = data.drop(['albumin'], axis = 1)
# data = data.drop(['dyspnoea'], axis = 1)
# data = data.drop(['total protein'], axis = 1)
# data = data.drop(['platelet count (PLT)'], axis = 1)
# data = data.drop(['duration of symptoms'], axis = 1)
# data = data.drop(['alkaline phosphatise (ALP)'], axis = 1)
# data = data.drop(['glucose'], axis = 1)
# data = data.drop(['pleural lactic dehydrogenise'], axis = 1)
# data = data.drop(['dead or not'], axis = 1)



# Get Top 10 OneR Features

# data = data.drop(['gender'], axis = 1)
# data = data.drop(['platelet count (PLT)'], axis = 1)
# data = data.drop(['C-reactive protein (CRP)'], axis = 1)
# data = data.drop(['cell count (WBC)'], axis = 1)
# data = data.drop(['city'], axis = 1)
# data = data.drop(['keep side'], axis = 1)
# data = data.drop(['alkaline phosphatise (ALP)'], axis = 1)
# data = data.drop(['cytology'], axis = 1)

# data = data.drop(['total protein'], axis = 1)
data = data.drop(['albumin'], axis = 1)
data = data.drop(['blood lactic dehydrogenise (LDH)'], axis = 1)
data = data.drop(['pleural protein'], axis = 1)
data = data.drop(['duration of asbestos exposure'], axis = 1)
data = data.drop(['ache on chest'], axis = 1)

data = data.drop(['pleural albumin'], axis = 1)
data = data.drop(['habit of cigarette'], axis = 1)
data = data.drop(['pleural level of acidity (pH)'], axis = 1)
data = data.drop(['white blood'], axis = 1)
data = data.drop(['age'], axis = 1)
data = data.drop(['dyspnoea'], axis = 1)
data = data.drop(['pleural effusion'], axis = 1)
data = data.drop(['weakness'], axis = 1)
data = data.drop(['asbestos exposure'], axis = 1)
data = data.drop(['type of MM'], axis = 1)
data = data.drop(['performance status'], axis = 1)
data = data.drop(['sedimentation'], axis = 1)

data = data.drop(['hemoglobin (HGB)'], axis = 1)

data = data.drop(['glucose'], axis = 1)
data = data.drop(['total protein'], axis = 1)

data = data.drop(['pleural lactic dehydrogenise'], axis = 1)
data = data.drop(['pleural glucose'], axis = 1)
data = data.drop(['dead or not'], axis = 1)
data = data.drop(['pleural thickness on tomography'], axis = 1)

# Get Top 10 Correlation Features


# data = data.drop(['gender'], axis = 1)
# data = data.drop(['cytology'], axis = 1)
# data = data.drop(['habit of cigarette'], axis = 1)
# data = data.drop(['ache on chest'], axis = 1)
# data = data.drop(['age'], axis = 1)
# data = data.drop(['dyspnoea'], axis = 1)
# data = data.drop(['pleural level of acidity (pH)'], axis = 1)
# data = data.drop(['pleural effusion'], axis = 1)
# data = data.drop(['pleural albumin'], axis = 1)
# data = data.drop(['C-reactive protein (CRP)'], axis = 1)
# data = data.drop(['duration of symptoms'], axis = 1)
# data = data.drop(['platelet count (PLT)'], axis = 1)
# data = data.drop(['duration of asbestos exposure'], axis = 1)
# data = data.drop(['weakness'], axis = 1)
# data = data.drop(['pleural thickness on tomography'], axis = 1)
# data = data.drop(['cell count (WBC)'], axis = 1)
# data = data.drop(['pleural protein'], axis = 1)
# data = data.drop(['type of MM'], axis = 1)
#
# data = data.drop(['city'], axis = 1)
# data = data.drop(['total protein'], axis = 1)
# data = data.drop(['blood lactic dehydrogenise (LDH)'], axis = 1)
# data = data.drop(['keep side'], axis = 1)
# data = data.drop(['white blood'], axis = 1)
# data = data.drop(['asbestos exposure'], axis = 1)
# data = data.drop(['performance status'], axis = 1)
# data = data.drop(['sedimentation'], axis = 1)
# data = data.drop(['hemoglobin (HGB)'], axis = 1)
# data = data.drop(['alkaline phosphatise (ALP)'], axis = 1)
# data = data.drop(['glucose'], axis = 1)
# data = data.drop(['pleural lactic dehydrogenise'], axis = 1)
# data = data.drop(['pleural glucose'], axis = 1)
# data = data.drop(['dead or not'], axis = 1)


# Get Top 10 Gain ratio Features

# data = data.drop(['type of MM'], axis = 1)
# data = data.drop(['keep side'], axis = 1)
# data = data.drop(['alkaline phosphatise (ALP)'], axis = 1)
# data = data.drop(['ache on chest'], axis = 1)
# data = data.drop(['habit of cigarette'], axis = 1)
# data = data.drop(['age'], axis = 1)
# data = data.drop(['gender'], axis = 1)
# data = data.drop(['cell count (WBC)'], axis = 1)
# data = data.drop(['platelet count (PLT)'], axis = 1)
# data = data.drop(['C-reactive protein (CRP)'], axis = 1)
# data = data.drop(['duration of symptoms'], axis = 1)
# data = data.drop(['duration of asbestos exposure'], axis = 1)
# data = data.drop(['cytology'], axis = 1)
# data = data.drop(['albumin'], axis = 1)
# data = data.drop(['dead or not'], axis = 1)
# data = data.drop(['pleural effusion'], axis = 1)
# data = data.drop(['pleural glucose'], axis = 1)
# data = data.drop(['pleural thickness on tomography'], axis = 1)
# data = data.drop(['city'], axis = 1)
#
# data = data.drop(['dyspnoea'], axis = 1)
# data = data.drop(['pleural level of acidity (pH)'], axis = 1)
# data = data.drop(['pleural albumin'], axis = 1)
# data = data.drop(['weakness'], axis = 1)
# data = data.drop(['pleural protein'], axis = 1)
# data = data.drop(['total protein'], axis = 1)
# data = data.drop(['blood lactic dehydrogenise (LDH)'], axis = 1)
# data = data.drop(['white blood'], axis = 1)
# data = data.drop(['asbestos exposure'], axis = 1)
# data = data.drop(['performance status'], axis = 1)
# data = data.drop(['sedimentation'], axis = 1)
# data = data.drop(['hemoglobin (HGB)'], axis = 1)
# data = data.drop(['glucose'], axis = 1)
# data = data.drop(['pleural lactic dehydrogenise'], axis = 1)


# Get Top 10 Info Gain  Features

# data = data.drop(['gender'], axis = 1)
# data = data.drop(['duration of symptoms'], axis = 1)
# data = data.drop(['platelet count (PLT)'], axis = 1)
# data = data.drop(['C-reactive protein (CRP)'], axis = 1)
# data = data.drop(['cell count (WBC)'], axis = 1)
# data = data.drop(['city'], axis = 1)
# data = data.drop(['keep side'], axis = 1)
# data = data.drop(['cytology'], axis = 1)
# data = data.drop(['alkaline phosphatise (ALP)'], axis = 1)

# data = data.drop(['habit of cigarette'], axis = 1)
# data = data.drop(['age'], axis = 1)
# data = data.drop(['ache on chest'], axis = 1)
# data = data.drop(['duration of asbestos exposure'], axis = 1)
# data = data.drop(['dyspnoea'], axis = 1)
# data = data.drop(['pleural level of acidity (pH)'], axis = 1)
# data = data.drop(['pleural effusion'], axis = 1)
# data = data.drop(['dead or not'], axis = 1)
# data = data.drop(['type of MM'], axis = 1)
# data = data.drop(['albumin'], axis = 1)
# data = data.drop(['pleural glucose'], axis = 1)
# data = data.drop(['pleural thickness on tomography'], axis = 1)
# data = data.drop(['pleural albumin'], axis = 1)
# data = data.drop(['weakness'], axis = 1)
# data = data.drop(['pleural protein'], axis = 1)
# data = data.drop(['total protein'], axis = 1)
# data = data.drop(['blood lactic dehydrogenise (LDH)'], axis = 1)
# data = data.drop(['white blood'], axis = 1)
# data = data.drop(['asbestos exposure'], axis = 1)
# data = data.drop(['performance status'], axis = 1)
# data = data.drop(['sedimentation'], axis = 1)
# data = data.drop(['hemoglobin (HGB)'], axis = 1)
# data = data.drop(['glucose'], axis = 1)
# data = data.drop(['pleural lactic dehydrogenise'], axis = 1)





print(data.info())
# # target class..
data['class of diagnosis'].value_counts()

from sklearn.model_selection import train_test_split
# train = data
# train = data.drop(['diagnosis method'], axis = 1)
# train = data.drop(['class of diagnosis'], axis = 1)


X = data.drop(['class of diagnosis'], axis = 1) # Features
y = data['class of diagnosis'] # Target variable







# split into 70:30 ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# describes info about train and test set
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)



# from catboost import CatBoostClassifier
# # evaluate the model
# model = CatBoostClassifier(verbose=0, n_estimators=100)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# # print classification report
# print(confusion_matrix(y_test,y_pred.round()))
# print(classification_report(y_test,y_pred.round()))


from sklearn.ensemble import GradientBoostingRegressor
# # # fit the model on the whole dataset
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#
# from xgboost import XGBClassifier
# model = XGBClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(confusion_matrix(y_test,y_pred.round()))
# print(classification_report(y_test,y_pred.round()))
#
#
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()

#Fitting model with trainig data
# model.fit(X_train, y_train)

import pickle
# save the model to disk
# filename = 'finalized_model.pkl'
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))



# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
#
# #Fitting model with trainig data
# regressor.fit(X, y)
#
# # Saving model to disk
# pickle.dump(regressor, open('model.pkl','wb'))


# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

# import joblib
# # save the model to disk
# filename = 'finalized_model.sav'
# joblib.dump(model, filename)

# some time later...

# load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, y_test)
# print(result)
# print('shakir')
# print(X_test)

# Ypredict = loaded_model.predict([[ -1.190760, 1.430058,  0.682688]])
# Ypredict = loaded_model.predict([[ 0.839799, -1.043281, -1.464797]])
# # Ypredict = loaded_model.predict(X_test)
# print(Ypredict)
