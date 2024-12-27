import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
# from numpy import sqrt
from numpy import abs
from pymongo import MongoClient
import requests
# from scrape_mlb_api import game_pk_pitch_data
from StrikeProbabilityModel import predict_strikes
from CatcherSLAAPercentiles import calculate_slaa_per_game_percentile
from CatcherSLAAPercentiles import color_by_percentile
import time

# Landscape A4
# Height
# Length
# 0,0 is top left

pdf = FPDF('L', 'in', 'Letter')
pdf.add_page()
pdf.set_font('Arial', size=30)
plt.rcParams['font.family'] = 'Arial'

def mongo_connect(cluster, date, player):
    if date == "all":
        db = cluster["Harbor_Hawks_24"]
        collection = db["Cape Trackman 24"]
        results = collection.find({"Catcher": player})
        results_list = list(results)
        df = pd.DataFrame(results_list)
        return df
    else:
        db = cluster["Harbor_Hawks_24"]
        collection = db["Cape Trackman 24"]
        results = collection.find({"Date": date, "Catcher": player})
        results_list = list(results)
        df = pd.DataFrame(results_list)
        return df

def read_tm_csv(filepath):
  df = pd.read_csv(filepath)
  return df

# StrikeZone and HomePlate, thank you Nick Wan
def draw_sz(sz_top=3.5, sz_bot=1.5, ls='k-'):
  """
  draw strike zone
  draw the strike zone on a plot using mpl
  inputs:
    sz_top: top of strike zone (ft)
    sz_bot: bottom of strike zone (ft)
    ls: linestyle (use `plt.plot()` linestyle conventions)
  output:
    strike zone plot
  """
  plt.plot([-0.708, 0.708], [sz_bot,sz_bot], ls)
  plt.plot([-0.708, -0.708], [sz_bot,sz_top], ls)
  plt.plot([0.708, 0.708], [sz_bot,sz_top], ls)
  plt.plot([-0.708, 0.708], [sz_top,sz_top], ls)
def draw_home_plate(catcher_perspective=True, ls='k-'):
  """
  draw home plate from either the catcher perspective or pitcher perspective
  inputs:
    catcher_perspective: orient home plate in the catcher POV. if False, orients
      home plate in the pitcher POV.
    ls: linestyle (use `plt.plot()` linestyle conventions)
  output:
    home plate plot
  """
  if catcher_perspective:
    plt.plot([-0.708, 0.708], [0,0], ls)
    plt.plot([-0.708, -0.708], [0,-0.3], ls)
    plt.plot([0.708, 0.708], [0,-0.3], ls)
    plt.plot([-0.708, 0], [-0.3, -0.6], ls)
    plt.plot([0.708, 0], [-0.3, -0.6], ls)
  else:
    plt.plot([-0.708, 0.708], [0,0], ls)
    plt.plot([-0.708, -0.708], [0,0.1], ls)
    plt.plot([0.708, 0.708], [0,0.1], ls)
    plt.plot([-0.708, 0], [0.1, 0.3], ls)
    plt.plot([0.708, 0], [0.1, 0.3], ls)
def draw_shadow_zone():
  """
  draw attack zones
  draw the statcast attack zones on a plot using mpl
  inputs:
    none! lulw
  output:
    attack zone plot
  """

  # outer heart / inner shadow
  plt.plot([-0.558, 0.558], [1.833,1.833], color=(227/255, 150/255, 255/255), ls='-', lw=2)
  plt.plot([-0.558, -0.558], [1.833,3.166], color=(227/255, 150/255, 255/255), ls='-', lw=2)
  plt.plot([0.558, 0.558], [1.833,3.166], color=(227/255, 150/255, 255/255), ls='-', lw=2)
  plt.plot([-0.558, 0.558], [3.166,3.166], color=(227/255, 150/255, 255/255), ls='-', lw=2)

  # outer shadow /  inner chase
  plt.plot([-1.108, 1.108], [1.166,1.166], color=(227/255, 150/255, 255/255), ls='-', lw=2)
  plt.plot([-1.108, -1.108], [1.166,3.833], color=(227/255, 150/255, 255/255), ls='-', lw=2)
  plt.plot([1.108, 1.108], [1.166,3.833], color=(227/255, 150/255, 255/255), ls='-', lw=2)
  plt.plot([-1.108, 1.108], [3.833,3.833], color=(227/255, 150/255, 255/255), ls='-', lw=2)

def get_metadata(df, date):
  opponent_replace = {
    "FAL_COM": "Falmouth",
    "COT_KET": "Cotuit",
    "BOU_BRA": "Bourne",
    "BRE_WHI": "Brewster",
    "CHA_ANG": "Chatham",
    "HAR_MAR": "Harwich",
    "ORL_FIR": "Orleans",
    "YAR_RED": "Yarmouth-Dennis",
    "WAR_GAT": "Wareham"
  }
  df['BatterTeam'] = df['BatterTeam'].replace(opponent_replace, regex=True)
  opponent = "vs " + df.loc[0, 'BatterTeam']
  catcher = df.loc[0, 'Catcher']
  if date == "all":
    date = "ALL GAMES"
    opponent = "Running Game, SLAA Stats are Averages"
  pdf.cell(3, .1)
  pdf.cell(w=4, h=0, txt=catcher, align='C', ln=1)
  pdf.set_font(family='Arial', size=16)
  pdf.cell(w=0, h=.7, txt=date, align='C', ln=1)
  pdf.cell(w=0, h=-.1, txt=opponent, align='C', ln=1)
  return date, catcher, opponent

def balls_strikes(df):
  called_pitches_df = df.loc[df['PitchCall'].isin(['StrikeCalled','BallCalled'])]
  called_pitches_df.reset_index(drop=True, inplace=True)
  balls_strikes = plt.figure(figsize=(3,3))
  balls_strikes = sns.scatterplot(x='PlateLocSide', y='PlateLocHeight', data=called_pitches_df, hue='PitchCall')
  draw_sz()
  draw_home_plate()
  draw_shadow_zone()
  plt.xlim(-3,3)
  plt.ylim(-1,5)
  plt.legend(loc="lower right",prop={'size': 8})
  plt.title('PitchCall (Catcher Perspective)')
  plt.savefig("balls_strikes.png", bbox_inches='tight')
  pdf.image('balls_strikes.png', .2, 1.2)
  plt.close()
  pdf.set_font(family='Arial', size=8)
  pdf.cell(1.92, 5.8)
  pdf.cell(.5, 2.70, "Strike Zone")
  pdf.set_font(family='Arial', size=8)
  pdf.set_text_color(227, 150, 255)
  pdf.cell(-.57, 3)
  pdf.cell(.5, 2.35, "Shadow Zone")

def running_game(df, date):
  catcher_throws_df = df.loc[df['PlayResult'].isin(['StolenBase','CaughtStealing'])]
  if not catcher_throws_df.empty:
    catcher_throws_df = catcher_throws_df.loc[:, ['Inning','PlayResult','ThrowSpeed', 'PopTime', 'ExchangeTime', 'BasePositionX', 'BasePositionZ']].copy()
    catcher_throws_df.rename(columns={'ExchangeTime':'Exchange'}, inplace=True)
    catcher_throws_df =  catcher_throws_df.dropna(how="all")
    catcher_throws_df['Base'] = np.where(
      (catcher_throws_df['BasePositionX'] < 100) & (catcher_throws_df['BasePositionZ'] < -30), '3B',
      np.where(
        (catcher_throws_df['BasePositionX'] < 100) & (catcher_throws_df['BasePositionZ'] > 30), '1B',
        '2B'
      )
    )
    replacements = {
      "CaughtStealing":"Caught",
      "StolenBase":"Steal"
    }
    catcher_throws_df['PlayResult'] = catcher_throws_df['PlayResult'].replace(replacements, regex=True)
    del catcher_throws_df['BasePositionX'], catcher_throws_df['BasePositionZ'] # not for display
    if date == "all":
      # 1. Delete the "Inning" column
      catcher_throws_df = catcher_throws_df.drop(columns=['Inning'])
      # 2. Remove rows where all three of ThrowSpeed, PopTime, and Exchange are NaN
      catcher_throws_df = catcher_throws_df.dropna(subset=['ThrowSpeed', 'PopTime', 'Exchange'], how='all')
      # 3. Group by PlayResult and Base and average the numeric columns
      catcher_throws_df = catcher_throws_df.groupby(['Base']).mean(numeric_only=True).round(2).reset_index()
      # # Filter rows for 2B and 3B
      # total_2B_rows = catcher_throws_df[catcher_throws_df['Base'] == '2B']
      # total_3B_rows = catcher_throws_df[catcher_throws_df['Base'] == '3B']
      # # Count "Caught" PlayResult for 2B and 3B
      # caught_2B_rows = total_2B_rows[total_2B_rows['PlayResult'] == 'Caught']
      # caught_3B_rows = total_3B_rows[total_3B_rows['PlayResult'] == 'Caught']
      # # Calculate percentages
      # caught_stealing_pct_2B = 100*(len(caught_2B_rows) / len(total_2B_rows)) if len(total_2B_rows) > 0 else 0
      # caught_stealing_pct_3B = 100*(len(caught_3B_rows) / len(total_3B_rows)) if len(total_3B_rows) > 0 else 0
      # pdf.set_font("Courier", "B", 30)
      # pdf.set_text_color(0,0,0)
      # pdf.set_xy(1, 7)
      # pdf.cell(2, 2, txt="2BCS%: " + str(caught_stealing_pct_2B), align='C', ln=1)
      # pdf.set_xy(3, 7)
      # pdf.cell(2, 2, txt="3BCS%: " + str(caught_stealing_pct_3B), align='C', ln=1)
    # Plotting the DataFrame as a table
  if not catcher_throws_df.empty:
    fig, ax = plt.subplots(figsize=(5.5, 1.2))  # You can adjust the size as needed
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    throws_table = ax.table(cellText=catcher_throws_df.values, colLabels=catcher_throws_df.columns, cellLoc='center', loc='center')
    throws_table.auto_set_font_size(False)
    throws_table.set_fontsize(9)
    throws_table.scale(1,1)  # Adjust the scale as needed
    plt.savefig('throws_table.png', bbox_inches='tight', pad_inches=.1)
    pdf.image("throws_table.png", .1, 6.2)
    plt.close()

def throw_locations(df):
  plt.rc('axes', labelsize=8)  # fontsize of the axes title
  catcher_throws_df = df.loc[df['PlayResult'].isin(['StolenBase','CaughtStealing'])]
  if not catcher_throws_df.empty:
    catcher_throws_df = catcher_throws_df.loc[:, ['Inning','PlayResult','ThrowSpeed', 'PopTime', 'ExchangeTime', 'BasePositionX', 'BasePositionZ', 'BasePositionY']].copy()
    catcher_throws_df.dropna(subset=['BasePositionX'], inplace=True) # remove all rows where BasePositionX, as it screws up categorization
    catcher_throws_df['base_stealing'] = np.where(
      (catcher_throws_df['BasePositionX'] < 100) & (catcher_throws_df['BasePositionZ'] < -30), '3B',
      np.where(
        (catcher_throws_df['BasePositionX'] < 100) & (catcher_throws_df['BasePositionZ'] > 30), '1B',
        '2B'
      )
    )
    steals_2b_df = catcher_throws_df.loc[catcher_throws_df['base_stealing'] == '2B']
    steals_2b = plt.figure(figsize=(1.5,.9))
    steals_2b = sns.scatterplot(x='BasePositionZ', y='BasePositionY', data=steals_2b_df)
    plt.xlim(-5, 5)
    plt.ylim(0,6)
    plt.title("Throws to 2B")
    plt.xlabel("<- 3B    1B ->")
    plt.ylabel('Height (ft)')
    plt.plot([-.625, -.625, .625, .625], [0, 5/12, 5/12, 0 ], c='black')
    plt.savefig("throw_locations_2b.png", bbox_inches='tight')
    pdf.image("throw_locations_2b.png", x=6.2, y=6.2)
    plt.close()

    steals_3b_df = catcher_throws_df.loc[catcher_throws_df['base_stealing'] == '3B'].copy()
    steals_3b_df['pythag_b'] = abs(steals_3b_df['BasePositionX'] - steals_3b_df['BasePositionZ'] - 127.279) / (np.sqrt(2))
    steals_3b_df['pythag_c'] = np.sqrt(
      (63.64 - steals_3b_df['BasePositionX']) ** 2 + (-63.64 - steals_3b_df['BasePositionZ']) ** 2)
    steals_3b_df['pythag_a'] = np.sqrt(steals_3b_df['pythag_c'] ** 2 - steals_3b_df['pythag_b'] ** 2)
    steals_3b = plt.figure(figsize=(1.5, .9))
    steals_3b = sns.scatterplot(x='pythag_a', y='BasePositionY', data=steals_3b_df)
    plt.xlim(-5, 5)
    plt.ylim(0, 6)
    plt.title("Throws to 3B")
    plt.xlabel("                     2B ->")
    plt.ylabel('Height (ft)')
    plt.plot([0, 0], [0, 6], c='gray', ls='--')
    plt.plot([0, 0, 1.25, 1.25], [0, 5/12, 5/12, 0 ], c='black')
    plt.savefig("throw_locations_3b.png", bbox_inches='tight')
    pdf.image("throw_locations_3b.png", x=8.5, y=6.2)
  plt.close()

def framing_table(df):
  # Define the boundaries of the shadow zone
  rect1_xmin, rect1_xmax = -0.558, 0.558
  rect1_ymin, rect1_ymax = 1.833, 3.166

  rect2_xmin, rect2_xmax = -1.108, 1.108
  rect2_ymin, rect2_ymax = 1.166, 3.833
  # Function to check if a point is in rectangle 2 but not in rectangle 1
  def in_shadow_zone(x, y):
    in_rect1 = rect1_xmin <= x <= rect1_xmax and rect1_ymin <= y <= rect1_ymax
    in_rect2 = rect2_xmin <= x <= rect2_xmax and rect2_ymin <= y <= rect2_ymax
    return 1 if in_rect2 and not in_rect1 else 0
  # Apply the function to each row in the dataframe
  df['InShadowZone'] = df.apply(lambda row: in_shadow_zone(row['PlateLocSide'], row['PlateLocHeight']), axis=1)
  framing_df = df.loc[df['PitchCall'].isin(['StrikeCalled', 'BallCalled']), ['TaggedPitchType', 'PitchCall', 'InShadowZone']]
  pitch_groups = {
    'Fast': ['Fastball', 'Four-Seam', 'Sinker'],
    'Breaking': ['Slider', 'Curveball', 'Cutter', 'Sweeper'],
    'Offspeed': ['ChangeUp', 'Splitter']
  }
  def map_pitch_group(pitch_type):
    for group, pitches in pitch_groups.items():
      if pitch_type in pitches:
        return group
    return 'Other'
  df['PitchGroup'] = df['TaggedPitchType'].apply(map_pitch_group)
  # Grouping and aggregating the data
  grouped_framing_df = df.groupby('PitchGroup').agg(
    Pitches=('TaggedPitchType', 'size'),
    ShadowPitches=('InShadowZone', lambda x: (x == 1).sum()),
    ShadowStrikes=('PitchCall', lambda x: ((x == 'StrikeCalled') & (df['InShadowZone'] == 1)).sum())
  ).reset_index()
  grouped_framing_df['ShadowStrike%'] = grouped_framing_df['ShadowStrikes'] / grouped_framing_df['ShadowPitches'] * 100
  grouped_framing_df['ShadowStrike%'] = grouped_framing_df['ShadowStrike%'].round(1)
  totals = pd.DataFrame([{
    'PitchGroup': 'Totals',
    'Pitches': grouped_framing_df['Pitches'].sum(),
    'ShadowPitches': grouped_framing_df['ShadowPitches'].sum(),
    'ShadowStrikes': grouped_framing_df['ShadowStrikes'].sum(),
    'ShadowStrike%': grouped_framing_df['ShadowStrikes'].sum() / grouped_framing_df['ShadowPitches'].sum() * 100 if grouped_framing_df[
                                                                                                                  'ShadowPitches'].sum() > 0 else float(
      'nan')
  }])
  totals['ShadowStrike%'] = totals['ShadowStrike%'].round(1)
  # Append totals to the dataframe
  grouped_framing_df = pd.concat([grouped_framing_df, totals], ignore_index=True)
  # unique_pitches = df['TaggedPitchType'].unique()
  # print("Unique pitch types:", unique_pitches)
  # other_pitches = df[df['PitchGroup'] == 'Other']['TaggedPitchType'].unique()
  # print("Pitches categorized as 'Other':", other_pitches)
  if not grouped_framing_df.empty:
    fig, ax = plt.subplots(figsize=(6, 1.5))  # You can adjust the size as needed
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    throws_table = ax.table(cellText=grouped_framing_df.values, colLabels=grouped_framing_df.columns, cellLoc='center', loc='center')
    throws_table.auto_set_font_size(False)
    throws_table.set_fontsize(9)
    throws_table.scale(1,1)  # Adjust the scale as needed
    plt.savefig('framing_table.png', bbox_inches='tight', pad_inches=0)
    pdf.image("framing_table.png", 4.3, 1.5)
    plt.close()

def strikes_above_average(df, date):
  if date == "all":
    minct = 6
  else:
    minct = 2
  prediction_labels, predicted_strike_prob = predict_strikes(df)
  df.loc[:, 'strike_probability'] = predicted_strike_prob
  called_pitches_df = df.loc[df['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
  called_pitches_df.reset_index(drop=True, inplace=True)
  called_pitches_df = called_pitches_df.copy()
  called_pitches_df.loc[:, 'Strikes Above Average'] = called_pitches_df['Strike'] - called_pitches_df[
    'strike_probability']
  strikes_above_average = plt.figure(figsize=(1.5, 1.5))
  norm = plt.Normalize(-.3, .3)
  plt.hexbin(x='PlateLocSide', y='PlateLocHeight', C='Strikes Above Average', data=called_pitches_df, cmap='coolwarm', gridsize=(10, 10), mincnt=minct, norm=norm)
  draw_sz()
  draw_home_plate()
  plt.xlim(-2, 2)
  plt.ylim(-1, 5)
  plt.colorbar()
  plt.title("Strikes Above Average", fontsize=10)
  plt.savefig("StrikesAboveAverage.png", bbox_inches="tight")
  pdf.image("StrikesAboveAverage.png", 4.3, 3)
  plt.close()
  strikes_above_average_total = called_pitches_df['Strikes Above Average'].sum()
  game_count = df['GameID'].nunique()
  if date == "all":
    strikes_above_average_total = strikes_above_average_total/game_count
  return strikes_above_average_total, called_pitches_df

def display_strikes_above_average(strikes_above_average_total):
  percentile = calculate_slaa_per_game_percentile(strikes_above_average_total)
  red, green, blue = color_by_percentile(percentile)
  pdf.set_font('Courier', size=45, style='B')
  pdf.set_text_color(red, green, blue)
  pdf.set_xy(7, 3.1)
  pdf.cell(2, 2, txt=str(round(strikes_above_average_total, 2)), align='C', ln=1)
  pdf.set_font('Arial', size=12)
  pdf.set_xy(7.1, 4.1)
  pdf.cell(2, 1, "Total Strikes Looking", align="C")
  pdf.set_xy(7.1, 4.3)
  pdf.cell(2, 1, "Above Average", align="C")
  pdf.set_font('Courier', size=45, style='B')
  pdf.set_xy(9, 3.1)
  pdf.cell(2, 2, txt=str(int(round(percentile, 0))), align='C', ln=1)
  pdf.set_font('Arial', size=12)
  pdf.set_xy(9, 4.1)
  pdf.cell(2, 1, "Percentile", align="C")
  pdf.set_xy(9, 4.3)
  pdf.cell(2, 1, "Total SLAA", align="C")

def lines_and_subtitles():
  pdf.line(0, 5.7, 11, 5.7)
  pdf.set_text_color(0, 0 ,0)
  pdf.set_font(family='Arial', style='B', size=20)
  pdf.set_fill_color(144, 238, 144)
  pdf.cell(w=-2.6, h=4.71, ln=1)
  pdf.cell(w=2.1, h=.4, txt="Running Game", fill=True, border=True, ln=1)

  pdf.line(0, 1.15, 11, 1.15)
  pdf.set_fill_color(240,128,128)
  pdf.set_xy(9,1.15)
  pdf.cell(w=1.2, h=.4, txt='Framing', fill=True, border=True)

def user_interface():
  catcher_name = input("Enter Player Name (Last, First): ")
  date = input("Enter Date (yyyy-mm-dd or 'all'): ")
  return catcher_name, date

def main():
  start_time = time.time()
  # df = read_tm_csv('/Users/quinnbooth/Desktop/Old Fun Projects/Hyannis 2024/cape_trackman_2024_full')
  cluster = MongoClient("mongodb+srv://HarborHawks:CCBL24champs!@harborhawks2024.yonksq3.mongodb.net/", serverSelectionTimeoutMS=50000,
                     socketTimeoutMS=50000,
                     connectTimeoutMS=50000)
  catcher_name, date = user_interface()
  df = mongo_connect(cluster, date, catcher_name)
  get_metadata(df, date)
  balls_strikes(df)
  running_game(df, date)
  throw_locations(df)
  framing_table(df)
  lines_and_subtitles()
  strikes_above_average_total, called_pitches_df = strikes_above_average(df, date)
  display_strikes_above_average(strikes_above_average_total)
  output_filename = f'/Users/quinnbooth/Desktop/Old Fun Projects/Hyannis 2024/CatcherReports/GeneratedPDFS/{catcher_name} Catcher Report, {date}.pdf'
  pdf.output(output_filename)
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Time taken to run: {elapsed_time} seconds")

main()

# TO DO:
  # Blocks
  # PyBaseball for MLB version

