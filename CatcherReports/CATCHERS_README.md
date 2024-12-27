README

**Overview**

This project models and visualizes defensive catcher performance, creating easily sharable and interpretable PDFs. 
The current model is trained on Cape League data, but in the future can be expanded to any league with Trackman or similar data. 
My next goal is to use the wealth of StatCast catching data to improve on these reports for MLB catchers.
Read more here: https://quinnbooth.substack.com/p/cape-league-catcher-reports

**Features**

  Visualization: Strike Zone, Shadow Zone, Strikes Looking Above Average, Throw Accuracy

  Metrics: Strikes Looking Above Average, Throw Speed, Pop Time, Transfer Time, Strikes Looking %

  PDF Reporting: Generate and save visualizations in a PDF report.

**Dependencies**

  pandas: Data manipulation and analysis.

  numpy: Numerical computations.

  matplotlib: Visualization and plotting.

  seaborn: Statistical data visualization.

  fpdf: PDF generation.

  pymongo: MongoDB interaction.

  requests: HTTP requests handling.

**File Structure**

  StrikeProbabilityModel.py : After feature selection, model type testing and training, the best model I had for strikes prediction.
  CatcherSLAAPercentiles.py : Scores catchers with percentiles, showing their rank among other Cape League catchers.
  ImprovedCatcherReports.py : Main script. Brings model and percentiles together along with graphs and generates the PDF.

**ImprovedCatcherReports.py Functions:**

  mongo_connect(cluster, date, player): Fetch data from Cape League MongoDB, created by Gabe Appelbaum.

  read_tm_csv(filepath): Read CSV files into a pandas DataFrame.

  draw_sz(sz_top, sz_bot, ls): Plot strike zone. (Thanks to Nick Wan)

  draw_home_plate(catcher_perspective, ls): Plot home plate in catcher or pitcher perspective. (Nick Wan)

  draw_shadow_zone(): Plot shadow zones. (Nick Wan)

  get_metadata(df, date): Extracts metadata from the dataset for PDF naming and titling.

  balls_strikes(df): Visualize pitch calls (balls and strikes).

  running_game(df, date): Analyzes and visualizes running game statistics.

  throw_locations(df): Visualizes catcher throw locations.

  framing_table(df): Splits by pitch type and shows Strike% and ShadowStrike% for each

  strikes_above_average(df, date): Calculates Strikes Looking Above Average stat. Takes predictions from model in StrikeProbabilityModel.py
  
  display_strikes_above_average(strikes_above_average_total): Visualizes hexbin plot of Strikes Above Average.

  lines_and_subtitles(): Places text and lines on PDF for organization and easier reading.

  user_interface(): Allows user to type in a catcher's name and a date to get their PDF. Name must be Last, First and Date is either yyyy-mm-dd or 'all'

