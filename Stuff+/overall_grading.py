import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

xgb_model_fast = joblib.load('xgb_stuff_plus_Fast_v1.pkl')
xgb_model_breaking = joblib.load('xgb_stuff_plus_Breaking_v1.pkl')
xgb_model_offspeed = joblib.load('xgb_stuff_plus_Offspeed_v1.pkl')

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
def prepare_features(trackman, heights):
    df = merge_datasets(trackman, heights)
    df = df.loc[df['Level'] == 'D1']
    df['height_in_feet'] = df['height'].apply(height_to_float)
    df = df[['Pitcher','TaggedPitchType','PitchCall','PitcherThrows','PitcherTeam','RelSpeed', 'InducedVertBreak',
             'HorzBreak', 'SpinRate', 'Extension', 'RelHeight', 'RelSide', 'height_in_feet']]
    df['CSW_bool'] = df['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging']).astype(int)
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

def feature_importance():
    importance = xgb_model_fast.feature_importances_
    # summarize feature importance
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

def grade_pitches(df, pitchgroup, team='all'):
    if team == 'all':
        df = df
    else:
        df = df.loc[df['PitcherTeam'] == team]
    stuff_df = pd.DataFrame(columns=['Pitcher','Team','PitcherThrows','Pitch','Pitches','RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'Extension', 'ArmAngle'])
    if pitchgroup == 'Fast':
        xgb_model = xgb_model_fast
    if pitchgroup == 'Breaking':
        xgb_model = xgb_model_breaking
    if pitchgroup == 'Offspeed':
        xgb_model = xgb_model_offspeed
    for pitcher in df['Pitcher'].unique():
        pitcher_df = df.loc[df['Pitcher'] == pitcher]
        pitcher_df.reset_index(inplace=True)
        throws = pitcher_df.loc[0, 'PitcherThrows']
        team = pitcher_df.loc[0, 'PitcherTeam']
        for pitch in df['TaggedPitchType'].unique():
            pitch_df = pitcher_df.loc[pitcher_df['TaggedPitchType'] == pitch, ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'Extension', 'ArmAngle']]
            pitch_df.dropna(inplace=True)
            if not pitch_df.empty:
                count = len(pitch_df)
                means = pitch_df.mean()
                means_df = means.to_frame().transpose()
                prediction = xgb_model.predict(means_df)
                # need to get the second value out of this prediction
                # divide by avg (need to get avg)
                means_df.loc[0, 'Predicted_Value'] = prediction
                means_df.loc[0, 'Pitcher'] = pitcher
                means_df.loc[0, 'Pitch'] = pitch
                means_df.loc[0, 'Pitches'] = count
                means_df.loc[0, 'PitcherThrows'] = throws
                means_df.loc[0, 'Team'] = team
                if not means_df.empty and means_df.notna().any().any():
                    stuff_df = pd.concat([stuff_df, means_df])
    stuff_df['HorzBreak'] = stuff_df.apply(
        lambda row: row['HorzBreak'] * -1 if row['PitcherThrows'] == 'Left' else row['HorzBreak'], axis=1) # convert back

    avg_value_fast = 0.005580696277320385
    avg_value_breaking = 0.026932841166853905
    avg_value_offspeed = 0.005580696277320385
    sd_value_fast = 0.026932841166853905
    sd_value_breaking = 0.005580696277320385
    sd_value_offspeed = 0.026932841166853905

    # stuff_df['Predicted_Value_Sign'] = np.where(stuff_df['Predicted_Value'] >= 0, 1, -1) # store the sign
    # stuff_df['Abs_Predicted_Value'] = np.abs(stuff_df['Predicted_Value']) # absolute value

    # AboveAvg Predicted_Value means a WORSE pitch. Means Above Avg runs given up
    if pitchgroup =='Fast':
        stuff_df['SDAboveAvg'] = (stuff_df['Predicted_Value'] - avg_value_fast)/sd_value_fast
    if pitchgroup =='Breaking':
        stuff_df['SDAboveAvg'] = (stuff_df['Predicted_Value'] - avg_value_breaking)/sd_value_breaking
    if pitchgroup =='Offspeed':
        stuff_df['SDAboveAvg'] = (stuff_df['Predicted_Value'] - avg_value_offspeed)/sd_value_offspeed

    # Now we want AboveAverage to mean better, or higher, Stuff+ so I SUBTRACT the number of SD from the mean of 100
    stuff_df['Stuff+'] = round(100 - stuff_df['SDAboveAvg']*10, 0)

    # Only want pitches with >= 10 thrown
    stuff_df = stuff_df.loc[stuff_df['Pitches'] >= 10]

    # Get displayable columns
    stuff_df = stuff_df[['Pitcher','Team', 'PitcherThrows', 'Pitch', 'Pitches', 'RelSpeed',
       'InducedVertBreak', 'HorzBreak', 'SpinRate', 'Extension', 'ArmAngle', 'Stuff+', 'Predicted_Value']]
    return stuff_df

def average_stuff(graded_fastballs, graded_breaking, graded_offspeed):
    # only pitches from a pitcher with >= 10 pitches in that type calculated in average!
    graded_pitches = pd.concat([graded_fastballs, graded_breaking, graded_offspeed])
    graded_pitches_mean = graded_pitches.loc[graded_pitches['Pitches'] >= 10,'Predicted_Value'].mean()
    graded_pitches_sd = graded_pitches.loc[graded_pitches['Pitches'] >= 10, 'Predicted_Value'].std()

    print(f'Average Value for All Pitch Type\n'
          f'{graded_pitches_mean}')

    print(f'Standard Deviation for All Pitch Type\n'
          f'{graded_pitches_sd}')

    return graded_pitches_mean, graded_pitches_sd
    #

def main():
    version = 1
    team = "SOU_TRO"
    trackman = pd.read_csv('/Users/quinnbooth/Desktop/USC Baseball/TM_2024_reg_szn (1).csv', low_memory=False)
    heights = pd.read_csv(
        '/Users/quinnbooth/Desktop/Old Fun Projects/Hyannis 2024/cape_model/college_batting_stats_2024_extra_info.csv')
    df_fast, df_breaking, df_offspeed = prepare_features(trackman, heights)
    graded_fastballs = grade_pitches(df_fast, 'Fast', team)
    graded_breaking = grade_pitches(df_breaking, 'Breaking', team)
    graded_offspeed = grade_pitches(df_offspeed, 'Offspeed', team)
    all_graded_pitches = pd.concat([graded_fastballs, graded_breaking, graded_offspeed])
    all_graded_pitches.to_csv(f'{team}_graded_pitches_v{str(version)}.csv')

main()