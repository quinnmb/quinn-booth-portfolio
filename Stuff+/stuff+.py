import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# What features?
    # RelSpeed
    # InducedVertBreak
    # HorzBreak
    # SpinRate
    # Extension
    # Arm Angle
        # Shoulder at x=0, z=70% * Player Height
        # RelSide
        # RelHeight

# Three different buckets
    # Fast
    # Breaking (includes cutter)
    # Offspeed

# Remember to flip left hand *-1

def reformat_name(name):
    parts = name.split()
    return f"{parts[-1]}, {parts[0]}"
def height_to_float(height_str):
    try:
        feet, inches = map(int, height_str.split('-'))
        return feet + inches / 12
    except (ValueError, AttributeError):
        return 6.17 # mean height weighted by pitches thrown (mean of height_in_feet column)
def merge_datasets(trackman, heights):
    heights['Pitcher'] = heights['Player'].apply(reformat_name)
    df = pd.merge(trackman, heights, on='Pitcher')
    return df
def map_pitch_group(pitch_type):
    pitch_groups = {
        'Fast': ['Fastball', 'Four-Seam', 'Sinker', 'Two-Seam'],
        'Breaking': ['Slider', 'Curveball', 'Cutter', 'Sweeper'],
        'Offspeed': ['ChangeUp', 'Splitter']
    }
    for group, pitches in pitch_groups.items():
        if pitch_type in pitches:
            return group
    return 'Other'

def prepare_target(trackman):
    count_transitions = pd.read_csv('count_transitions.csv')
    trackman['PitchCall'] = trackman['PitchCall'].replace('FoulBall', 'FoulBallNotFieldable')
    trackman['PitchCall'] = trackman['PitchCall'].replace('BallinDirt', 'BallCalled')
    trackman['PitchResult'] = trackman.apply(lambda row: row['PitchCall'] if row['PlayResult'] == 'Undefined' else row['PlayResult'], axis=1)
    values_to_remove = ['BallIntentional', 'CaughtStealing', 'StolenBase']
    trackman = trackman[~trackman['PitchResult'].isin(values_to_remove)]
    merged_trackman = trackman.merge(count_transitions[['Balls', 'Strikes', 'PitchResult', 'Pitch_Value']],
                               on=['Balls', 'Strikes', 'PitchResult'],
                               how='left')
    na_pitch_value_df = merged_trackman[merged_trackman['Pitch_Value'].isna()]
    merged_trackman.dropna(subset=['Pitch_Value'], inplace=True)
    return merged_trackman

def prepare_features(trackman, heights):
    df = merge_datasets(trackman, heights)
    df = df.loc[df['Level'] == 'D1']
    df['height_in_feet'] = df['height'].apply(height_to_float)
    df = df[['Pitcher','TaggedPitchType','PitchCall','PitcherThrows','PitcherTeam','RelSpeed', 'InducedVertBreak',
             'HorzBreak', 'SpinRate', 'Extension', 'RelHeight', 'RelSide', 'height_in_feet', 'Pitch_Value']]
    df['HorzBreak'] = df.apply(
        lambda row: row['HorzBreak'] * -1 if row['PitcherThrows'] == 'Left' else row['HorzBreak'], axis=1)
    df['RelSide'] = df.apply(
        lambda row: row['RelSide'] * -1 if row['PitcherThrows'] == 'Left' else row['RelSide'], axis=1)
    df['ShoulderHeight'] = (.7)*df['height_in_feet'] # estimation I saw on Twitter
    df['ArmAngle'] = df.apply(
        lambda row: np.degrees(np.arctan2(row['RelSide'], row['RelHeight'] - row['ShoulderHeight'])), axis=1)
    df.dropna(inplace=True)
    df['PitchGroup'] = df['TaggedPitchType'].apply(map_pitch_group)
    df_fast = df.loc[df['PitchGroup'] == 'Fast']
    df_breaking = df.loc[df['PitchGroup'] == 'Breaking']
    df_offspeed = df.loc[df['PitchGroup'] == 'Offspeed']
    return df_fast, df_breaking, df_offspeed

def X_y_splits(df_pitchgroup):
    X = df_pitchgroup[['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'Extension', 'ArmAngle']]
    y = df_pitchgroup['Pitch_Value']
    X_train_pitchgroup, X_test_pitchgroup, y_train_pitchgroup, y_test_pitchgroup = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train_pitchgroup, X_test_pitchgroup, y_train_pitchgroup, y_test_pitchgroup

def train_model(X_train_pitchgroup, y_train_pitchgroup):
    # bagging = BaggingRegressor(n_estimators=30, random_state=1)
    # bagging.fit(X_train_pitchgroup, y_train_pitchgroup)
    #
    # rf = RandomForestRegressor(n_estimators=30, random_state=1)
    # rf.fit(X_train_pitchgroup, y_train_pitchgroup)

    xgb = XGBRegressor(n_estimators=100, random_state=1)
    xgb.fit(X_train_pitchgroup, y_train_pitchgroup)

    return xgb

def test_model(model, X_test_pitchgroup,y_test_pitchgroup, pitchgroup, accuracy_df, model_str):
    y_predictions = model.predict(X_test_pitchgroup)
    mse_test = mean_squared_error(y_test_pitchgroup, y_predictions)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test_pitchgroup, y_predictions)

    new_results = pd.DataFrame({
        'Model': [model_str],
        'Dataset': [pitchgroup],  # Add dataset name for each model
        'MSE Test': [mse_test],
        'RMSE Test': [rmse_test],
        'MAE Test': [mae_test]
    })
    accuracy_df = pd.concat([accuracy_df, new_results])
    return accuracy_df

def create_score_models(df_pitchgroup, pitchgroup, accuracy_df):
    version = 1
    X_train, X_test, y_train, y_test = X_y_splits(df_pitchgroup)
    xgb = train_model(X_train, y_train)
    # accuracy_df = test_model(bagging, X_test, y_test, pitchgroup, accuracy_df, 'Bagging')
    # accuracy_df = test_model(rf, X_test, y_test, pitchgroup, accuracy_df, 'Random Forest')
    accuracy_df = test_model(xgb, X_test, y_test, pitchgroup, accuracy_df, 'XGB')
    joblib.dump(xgb, f'xgb_stuff_plus_{pitchgroup}_v{str(version)}.pkl')
    return accuracy_df

def main():
    trackman = pd.read_csv('/Users/quinnbooth/Desktop/USC Baseball/TM_2024_reg_szn (1).csv', low_memory=False)
    heights = pd.read_csv('/Users/quinnbooth/Desktop/Old Fun Projects/Hyannis 2024/cape_model/college_batting_stats_2024_extra_info.csv')
    merged_trackman = prepare_target(trackman)
    df_fast, df_breaking, df_offspeed = prepare_features(merged_trackman, heights)
    accuracy_df = pd.DataFrame({},
                               columns=['Model','Dataset', 'MSE Test', 'RMSE Test', 'MAE Test'])
    accuracy_df = create_score_models(df_fast, 'Fast', accuracy_df)
    accuracy_df = create_score_models(df_breaking, 'Breaking', accuracy_df)
    accuracy_df = create_score_models(df_offspeed, 'Offspeed', accuracy_df)
    # XGB is best for all three!
    print(accuracy_df.to_string())

main()