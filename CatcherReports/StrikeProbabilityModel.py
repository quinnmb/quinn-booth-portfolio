import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
# from PythonMongoDB import mongo_connect
# # Method Using this year and last year's data:
# df1 = pd.read_csv('/Users/quinnbooth/Desktop/Hyannis 2024/Cape_Database_Fixed_Names_Final.csv', low_memory=False)
# df2 = pd.read_csv('/Users/quinnbooth/Desktop/Hyannis 2024/combined_data.csv', low_memory=False)
# df = pd.concat([df1, df2])  # combining all of last year and all data from this year into one big dataframe

# Method using just this year. This feels like a good switch. Similar accuracy
# df = mongo_connect()
df = pd.read_csv('/Users/quinnbooth/Desktop/Old Fun Projects/Hyannis 2024/cape_trackman_2024_full', low_memory=False)
called_pitches_df = df.loc[df['PitchCall'].isin(['BallCalled', 'StrikeCalled'])].copy()
called_pitches_df.loc[:, 'Strike'] = np.where(called_pitches_df.loc[:, 'PitchCall'] == 'StrikeCalled', 1, 0)
called_pitches_df.loc[:, 'BatterRighty'] = np.where(called_pitches_df.loc[:, 'BatterSide'] == 'Right', 1, 0)
X = called_pitches_df[['PlateLocHeight', 'PlateLocSide', 'BatterRighty']]
y = called_pitches_df['Strike']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=1, objective='binary:logistic', random_state=1)
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_prediction_probs = model.predict_proba(X_train)
test_prediction_probs = model.predict_proba(X_test)
# print("Precision = {}".format(precision_score(y_test, test_predictions, average='macro')))
# print("Recall = {}".format(recall_score(y_test, test_predictions, average='macro')))
# print("Accuracy = {}".format(accuracy_score(y_test, test_predictions)))

def predict_strikes(tm_df):
    tm_df.loc[:, 'BatterRighty'] = np.where(tm_df.loc[:, 'BatterSide'] == 'Right', 1, 0).copy()
    tm_df.loc[:, 'Strike'] = np.where(tm_df.loc[:, 'PitchCall'] == 'StrikeCalled', 1, 0)
    X_data = tm_df.loc[:, ['PlateLocHeight', 'PlateLocSide', 'BatterRighty']]
    prediction_labels = model.predict(X_data)
    prediction_proba = model.predict_proba(X_data)
    predicted_strike_prob = prediction_proba[:, 1]
    return prediction_labels, predicted_strike_prob
