import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from pymongo import MongoClient
import time
from datetime import datetime

plt.rcParams['font.family'] = 'Arial'

def mongo_connect(cluster, team):
    db = cluster["Harbor_Hawks_24"]
    collection = db["Cape Trackman 24"]
    results = collection.find({"BatterTeam":team})
    results_list = list(results)
    df = pd.DataFrame(results_list)
    return df

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
def draw_home_plate(catcher_perspective=False, ls='k-'):
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

def new_columns(df):
    # for continuity purposes
    df.sort_values(by=['Date','PitchNo'],ascending=[True, True], inplace=True)

    # PA, start, end
    df['StartofPA'] = np.where(df['PitchofPA'] == 1, 1, 0)
    df['EndofPA'] = np.where((df['PitchofPA'].shift(-1) == 1), 1, 0)

    # wOBA
    conditions = [
        (df['PlayResult'] == 'Single'),
        (df['PlayResult'] == 'Double'),
        (df['PlayResult'] == 'Triple'),
        (df['PlayResult'] == 'HomeRun')
    ]
    wOBAcon_weights = [.897, 1.23, 1.59, 1.97]
    df['wOBA'] = np.select(conditions, wOBAcon_weights, default=0)

    df['Contact'] = np.where(df['PitchCall'] == 'InPlay', 1, 0)

    # pitch grouping
    pitch_groups = {
        'Fastball': ['Fastball', 'Four-Seam', 'Sinker'],
        'SL+CT': ['Slider', 'Cutter', 'Sweeper'],
        'CB': ['Curveball'],
        'CH+SP': ['ChangeUp', 'Splitter'],
    }
    def map_pitch_group(pitch_type):
        for group, pitches in pitch_groups.items():
            if pitch_type in pitches:
                return group
        return 'Other'
    df['PitchGroup'] = df['TaggedPitchType'].apply(map_pitch_group)

    # in zone
    df['InZone'] = np.where((df['PlateLocSide'].between(-0.708, 0.708) & df['PlateLocHeight'].between(1.5, 3.5)), 1, 0)

    # Swing
    df['Swing'] = np.where(df['PitchCall'].isin(['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable']), 1, 0)
    return df

def wOBAcon(df, pdf):
    contact_df = df.loc[df['PitchCall'] == 'InPlay']
    plt.figure(figsize=(3, 3))
    norm = plt.Normalize(0, .6)
    hexplot = plt.hexbin(x='PlateLocSide', y='PlateLocHeight', C='wOBA', data=contact_df, cmap='coolwarm',
               gridsize=4, mincnt=1, norm=norm)
    plt.xlim(-2, 2)
    plt.ylim(-1, 5)
    draw_sz()
    draw_home_plate()
    plt.colorbar()
    plt.title("Damage")
    plt.savefig("wOBAcon.png", bbox_inches='tight')
    pdf.image("wOBAcon.png", 2.8, .5)
    plt.close()

def whiffs(df, pdf):
    whiffs_df = df.loc[df['PitchCall'] == 'StrikeSwinging']
    plt.figure(figsize=(3, 3))
    norm = plt.Normalize(0, 10)
    plt.hexbin(data=whiffs_df, x='PlateLocSide', y='PlateLocHeight', cmap='Reds', gridsize=4, mincnt=1, norm=norm)
    plt.xlim(-2, 2)
    plt.ylim(-1, 5)
    draw_sz()
    draw_home_plate()
    plt.colorbar()
    plt.title("Whiffs Frequency")
    plt.savefig("Whiffs.png", bbox_inches='tight')
    pdf.image("Whiffs.png", 6.8, .5)
    plt.close()

def pitch_type_table(df, pdf):
    def whiff_percentage(group):
        return 100 * (df.loc[group.index,'PitchCall'] == 'StrikeSwinging').sum() / (df.loc[group.index, 'Swing'] == 1).sum()

    def iz_whiff_percentage(group):
        return 100 * ((df.loc[group.index, 'PitchCall'] == 'StrikeSwinging') & (
                    df.loc[group.index, 'InZone'] == 1)).sum() / (df.loc[group.index, 'Swing'] == 1).sum()

    def chase_percentage(group):
        return 100 * ((df.loc[group.index, 'Swing'] == 1) & (df.loc[group.index, 'InZone'] == 0)).sum() / (df.loc[group.index, 'InZone'] == 0).sum()

    contact_df = df.loc[df['PitchCall'] == 'InPlay']

    grouped_contact_df = contact_df.groupby('PitchGroup').agg(
        wOBAcon=('wOBA', 'mean'),
    )

    grouped_df = df.groupby('PitchGroup').agg(
        Pitches=('PitchGroup', 'count'),
        Whiff_pct=('PitchGroup', whiff_percentage),
        IZWhiff_pct=('PitchGroup', iz_whiff_percentage),
        Chase_pct=('Swing', chase_percentage)
    )

    # Calculate totals manually
    total_contact = pd.Series({
        'wOBAcon': contact_df['wOBA'].mean()
    }, name='Totals')

    total_group = pd.Series({
        'Pitches': df['PitchGroup'].count(),
        'Whiff_pct': whiff_percentage(df),
        'IZWhiff_pct': iz_whiff_percentage(df),
        'Chase_pct': chase_percentage(df)
    }, name='Totals')

    # Convert totals to DataFrame
    total_contact_df = total_contact.to_frame().T
    total_group_df = total_group.to_frame().T

    # Append totals to grouped dataframes
    grouped_contact_df = pd.concat([grouped_contact_df, total_contact_df])
    grouped_df = pd.concat([grouped_df, total_group_df])

    # Set the order of rows
    pitch_group_order = ['Fastball', 'SL+CT', 'CB', 'CH+SP','Other', 'Totals']
    grouped_contact_df = grouped_contact_df.reindex(pitch_group_order)
    grouped_df = grouped_df.reindex(pitch_group_order)

    # combine two tables
    combined_df = pd.concat([grouped_contact_df, grouped_df], axis=1)

    # Round the wOBAcon column to three decimal places
    combined_df['wOBAcon'] = combined_df['wOBAcon'].round(3)

    # Round all other values to one decimal place
    combined_df = combined_df.round({col: 1 for col in combined_df.columns if col != 'wOBAcon'})

    # display
    if not combined_df.empty:
        fig, ax = plt.subplots(figsize=(6, 1.5))  # You can adjust the size as needed
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        pitch_type_table = ax.table(cellText=combined_df.values, colLabels=combined_df.columns, rowLabels=combined_df.index
                                ,cellLoc='center', loc='center')
        pitch_type_table.auto_set_font_size(False)
        pitch_type_table.set_fontsize(12)
        pitch_type_table.scale(1, 1)  # Adjust the scale as needed
        plt.savefig('pitch_type_table.png', bbox_inches='tight', pad_inches=0)
        pdf.image("pitch_type_table.png", 3.1, 5)
        plt.close()

def basic_stats(df, pdf):
    basic_stats = pd.DataFrame(columns=['G','PA', 'XBH', 'HR', 'K', 'BB', 'K%', 'BB%'])
    G = (df['GameID'].nunique())
    PA = (df['StartofPA']).sum()
    XBH = (df['PlayResult'].isin(['Double', 'Triple', 'HomeRun'])).sum()
    HR = (df['PlayResult'] == 'HomeRun').sum()
    K = (df['KorBB'] == 'Strikeout').sum()
    BB = (df['KorBB'] == 'Walk').sum()
    K_pct = 100*K/PA
    BB_pct = 100*BB/PA
    basic_stats.loc[0] = [G, PA, XBH, HR, K, BB, K_pct, BB_pct]
    basic_stats = basic_stats.round(1)
    if not basic_stats.empty:
        fig, ax = plt.subplots(figsize=(4,.5))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        basic_stats_table = ax.table(cellText=basic_stats.values, colLabels=basic_stats.columns,
                                cellLoc='center', loc='center')
        basic_stats_table.auto_set_font_size(False)
        basic_stats_table.set_fontsize(12)
        basic_stats_table.scale(1, 1)  # Adjust the scale as needed
        plt.savefig("basic_stats_table.png", bbox_inches='tight', pad_inches=0)
        pdf.image("basic_stats_table.png", 5, 7)
        plt.close()

def header(df, pitcher_hand, pdf):
    df.reset_index(drop=True, inplace=True)
    if not df.empty:
        batter_name = df.loc[0, 'Batter']
    else:
        batter_name = "????"
    unique_sides = df['BatterSide'].unique() # decides switch or just plain left, right
    if len(unique_sides) == 1:
        batter_hand = unique_sides[0]
    else:
        batter_hand = 'Switch'
    today_date = datetime.today().strftime('%Y-%m-%d')
    pdf.set_xy(1, .5)
    pdf.set_font(family='Arial', size=16)
    pdf.cell(1, .5, txt=batter_name,align='C', ln=2)
    pdf.cell(1, .5, txt="Batter Hand: "+ batter_hand,align='C', ln=2)
    pdf.cell(1, .5, txt="Pitcher Hand: " + pitcher_hand, align='C', ln=2)
    pdf.set_font(family='Arial', size=10)
    pdf.cell(1, .5, txt="Data Pulled: "+today_date, align='C', ln=2)

def user_interface():
    team = input("Enter team code, e.g HYA_HAR: ")
    return team

def main():
    cluster = MongoClient("mongodb+srv://HarborHawks:CCBL24champs!@harborhawks2024.yonksq3.mongodb.net/",
                          serverSelectionTimeoutMS=50000,
                          socketTimeoutMS=50000,
                          connectTimeoutMS=50000)
    team = user_interface()
    df = mongo_connect(cluster, team)
    batters_list = df['Batter'].unique().tolist()
    for batter in batters_list:
        df_batter = df.loc[df['Batter'] == batter] # individual batter dataframe
        for pitcher_hand in ['Left', 'Right']: # split by side
            pdf = FPDF('L', 'in', 'Letter')
            pdf.add_page()
            df_batter_hand = df_batter.loc[df_batter['PitcherThrows'] == pitcher_hand]
            df_batter_hand = new_columns(df_batter_hand)
            wOBAcon(df_batter_hand, pdf)
            whiffs(df_batter_hand, pdf)
            pitch_type_table(df_batter_hand, pdf)
            basic_stats(df_batter_hand, pdf)
            header(df_batter_hand, pitcher_hand, pdf)
            # time.sleep(5)
            pdf.output(f"/Users/quinnbooth/Desktop/Hyannis 2024/HitterReports/GeneratedPDFs/{batter} Report vs {pitcher_hand}.pdf")
            # time.sleep(5)

main()

# at the end: user interface where you can input team and
# for unique player values of that team it goes through the whole process, generating folders and the pdfs to put inside them. Can remove dead players later

# Base stats table
    # G
    # PA
    # XBH
    # HR
    # K
    # BB
    # K%
    # BB%
# Header - name, batter hand, filter(pitcher) hand, team, data pulled date

# wOBAcon heatmap DONE
# wOBAcon by pitch type DONE
# Whiff Heatmap DONE
# Whiff, IZWhiff, Chase table by pitch type DONE

