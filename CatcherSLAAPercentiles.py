import pandas as pd
from pymongo import MongoClient
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from StrikeProbabilityModel import predict_strikes
from datetime import datetime
# Re-calculated percentiles on 8-14-2024

def mongo_connect(cluster): # Connects to database
  db = cluster["Harbor_Hawks_24"]
  collection = db["Cape Trackman 24"]
  results = collection.find({})
  results_list = list(results)
  df = pd.DataFrame(results_list)
  return df

def download_trackman_data(df): # I don't want to get the whole Mongo DF each time for a report, so I will download it periodically and pass it as a CSV
    today_date = datetime.today().strftime('%Y-%m-%d')
    filename = f"catchers_thru_{today_date}.csv"
    prediction_labels, predicted_strike_prob = predict_strikes(df)
    df.loc[:, 'strike_probability'] = predicted_strike_prob
    called_pitches_df = df.loc[df['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
    called_pitches_df.reset_index(drop=True, inplace=True)
    called_pitches_df = called_pitches_df.copy()
    called_pitches_df.loc[:, 'Strikes Above Average'] = called_pitches_df['Strike'] - called_pitches_df[
        'strike_probability']
    grouped_df = called_pitches_df.groupby('Catcher').agg(
        Strikes_Above_Average_Sum=('Strikes Above Average', 'sum'),
        GameCount=('GameID', 'nunique')
    ).reset_index()
    grouped_df_filtered = grouped_df.loc[grouped_df['GameCount'] > 1].copy()
    grouped_df_filtered['Strikes_Above_Average_Per_Game'] = grouped_df_filtered['Strikes_Above_Average_Sum'] / grouped_df_filtered['GameCount']
    grouped_df_filtered.sort_values(by="Strikes_Above_Average_Per_Game", ascending=False, ignore_index=True).to_csv(filename)
    return filename

def strikes_above_average(df): # function similar to ImprovedCatcherReports.py. Could not import due to loop concerns
  prediction_labels, predicted_strike_prob = predict_strikes(df)
  df.loc[:, 'strike_probability'] = predicted_strike_prob
  called_pitches_df = df.loc[df['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
  called_pitches_df.reset_index(drop=True, inplace=True)
  called_pitches_df = called_pitches_df.copy()
  called_pitches_df.loc[:, 'Strikes Above Average'] = called_pitches_df['Strike'] - called_pitches_df[
    'strike_probability']
  strikes_above_average_total = called_pitches_df['Strikes Above Average'].sum()
  return strikes_above_average_total, called_pitches_df

# OLD STUFF GETTING CATCHER GAMES
# def split_dataframe_by_gameid_and_pitcherteam(df):
#     # Get unique combinations of GameID and PitcherTeam
#     unique_combinations = df[['GameID', 'PitcherTeam']].drop_duplicates()
#
#     # Create a dictionary to hold the individual dataframes
#     split_dfs = {}
#
#     # Loop through each unique combination and create a new dataframe
#     for _, row in unique_combinations.iterrows():
#         game_id = row['GameID']
#         pitcher_team = row['PitcherTeam']
#         game_df = df[(df['GameID'] == game_id) & (df['PitcherTeam'] == pitcher_team)]
#         split_dfs[f'game_{game_id}_team_{pitcher_team}'] = game_df
#
#     return split_dfs

# def calculate_each_game_slaa(huge_df):
#     split_dfs = split_dataframe_by_gameid_and_pitcherteam(huge_df)
#     slaa_list = []
#     for game_id_team, game_df in split_dfs.items():
#         result = strikes_above_average(game_df)
#         slaa_list.append(result)
#         print("Appended SLAA for", game_id_team, "    Result:", result)
#     return slaa_list
#
# def save_slaa_list(slaa_list):
#     game_slaa_df = pd.DataFrame({'Result': slaa_list})
#     game_slaa_df.to_csv('game_slaa_df.csv', index=False)
#     print("Results exported to game_slaa_df.csv")



# def calculate_percentile(slaa): # This is the old way I did percentiles, getting all catcher games.
# # I like it better doing everything on a per game basis.
#     game_slaa_df = pd.read_csv('game_slaa_df.csv')
#     percentile = percentileofscore(game_slaa_df['Result'], slaa, kind='weak')
#     return percentile

def calculate_slaa_per_game_percentile(slaa, filename='/Users/quinnbooth/Desktop/Old Fun Projects/Hyannis 2024/CatcherReports/catchers_thru_2024-08-14.csv'):
    catchers_slaa_df = pd.read_csv(filename)
    catchers_slaa_df = catchers_slaa_df.loc[catchers_slaa_df['GameCount'] > 1]
    per_game_percentile = percentileofscore(catchers_slaa_df['Strikes_Above_Average_Per_Game'], slaa, kind='weak')
    return per_game_percentile

def color_by_percentile(percentile):
    if percentile < 10:
        # Very dark red
        return 150, 0, 0  # RGB values for very dark red
    elif percentile < 20:
        # Slightly less dark red
        return 175, 0, 0  # RGB values for slightly less dark red
    elif percentile < 30:
        # Slightly less dark red
        return 200, 0, 0  # RGB values for slightly less dark red
    elif percentile < 40:
        # Less dark red
        return 225, 0, 0  # RGB values for less dark red
    elif percentile < 60:
        # Plain black
        return 0, 0, 0  # RGB values for black
    elif percentile < 70:
        # Light green
        return 0, 225, 0  # RGB values for light green
    elif percentile < 80:
        # Light green
        return 0, 200, 0  # RGB values for light green
    elif percentile < 90:
        # Dark green
        return 0, 175, 0  # RGB values for dark green
    else:
        # Very dark green
        return 0, 150, 0  # RGB values for very dark green

def main():
    cluster = MongoClient("mongodb+srv://HarborHawks:CCBL24champs!@harborhawks2024.yonksq3.mongodb.net/")
    df = mongo_connect(cluster)
    filename = download_trackman_data(df)
    strikes_above_average_total, full_called_strikes_df = strikes_above_average(df)

# main()


# percentiles for all cape only when catcher has caught > 5 games