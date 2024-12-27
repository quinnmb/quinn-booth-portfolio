README

**Overview:**

This script generates PDF reports for baseball hitters based on their performance data for scouting and pitch planning purposes. 
I created these reports direct to coach requests late in the Cape season, integrating all charts and stats they said they needed.
It extracts data from a MongoDB collection or Trackman file, processes it to compute statistics, and visualizes key metrics as heatmaps and tables. 
Each report is split by the pitcher's throwing hand (left or right) in order to best create pitch plans.

**Features**

  MongoDB Integration: Connects to a MongoDB collection and retrieves hitter data for a specified team. Can be easily modified to accept a Trackman file rather than MongoDB collection.

  PDF Report Generation: Creates PDF reports for each hitter, split by the pitcher's throwing hand.

**Data Visualization:**

  wOBAcon Heatmap: Visualizes weighted On-Base Average (wOBA) for balls in play.

  Whiff Heatmap: Highlights the frequency of swings and misses.

**Performance Tables:**

  Basic stats table: Includes games, plate appearances, extra-base hits, home runs, strikeouts, walks, and percentages.

  Pitch type table: Provides wOBAcon, whiff rates, in-zone whiff rates, and chase rates by pitch type.

  Header Information: Displays hitter name, batter hand, pitcher hand, team code, and data pull date.

**Python libraries:** pandas, numpy, matplotlib, seaborn, fpdf, pymongo

**Functions:**

  mongo_connect(cluster, team): Connects to MongoDB and retrieves data for the specified team.

  draw_sz(sz_top=3.5, sz_bot=1.5, ls='k-'): Draws the strike zone on a plot. (Credit to Nick Wan)

  draw_home_plate(catcher_perspective=False, ls='k-'): Draws the home plate from the perspective of the pitcher or catcher. (Nick Wan)

  new_columns(df): Adds derived columns to the DataFrame for processing, including StartofPA, EndofPA, wOBA, Contact, PitchGroup, InZone, and Swing.

  wOBAcon(df, pdf): Creates a heatmap for wOBA of balls in play and adds it to the PDF.

  whiffs(df, pdf): Generates a heatmap of whiffs and adds it to the PDF.

  pitch_type_table(df, pdf): Generates a table of pitch statistics, including wOBAcon, whiff rates, in-zone whiff rates, and chase rates, and adds it to the PDF.

  basic_stats(df, pdf): Creates a table of basic stats (games, plate appearances, extra-base hits, etc.) and adds it to the PDF.

  header(df, pitcher_hand, pdf): Adds a header to the PDF with hitter name, batter hand, pitcher hand, and data pull date.

  user_interface(): Prompts the user to input a team code.

**Output**

PDF files for each hitter in the format:

[Hitter Name] Report vs [Pitcher Hand].pdf


