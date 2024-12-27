README

**Overview:**
This project builds a machine learning model to predict the value of a pitch (Pitch_Value) based on Trackman and pitcher data. 
It organizes pitches into three categories—Fast, Breaking, and Offspeed to compare/regress pitches to their similar type.

**File Overview:**
- stuff+.py : Trains the model. I tested with several different feature combinations and model types before tuning the model for best possible performance.
- overall_grading.py : This script takes the model Run Value predictions and uses them to grade each NCAA D1 pitcher's offerings at the 100 scale.

**Libraries Used**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
  - BaggingRegressor
  - RandomForestRegressor
- xgboost
  - XGBRegressor
- joblib

**Features Used**
- RelSpeed
- InducedVertBreak
- HorzBreak
- SpinRate
- Extension
- Arm Angle
  - The shoulder is estimated at z = 70% * Player Height.

**Preprocessing Notes**:
Left-handed pitchers’ data is flipped (*-1) for HorzBreak and RelSide to align with right-handed pitchers.
This way, we have more data and lefties and righties are judged the same way.

**Pitch Buckets**
Fast: Includes pitches like fastballs, four-seamers, sinkers, and two-seamers.
Breaking: Includes sliders, curveballs, cutters, and sweepers.
Offspeed: Includes changeups and splitters.

**Functions:**

  reformat_name(name): Reformats player names from "First Last" to "Last, First". Important for future data merging.
  
  height_to_float(height_str): Converts height in "feet-inches" (e.g., 6-2) to a float value in feet. Returns the mean height if invalid.
  
  merge_datasets(trackman, heights): Merges Trackman and pitcher height datasets, using the above two functions.

  map_pitch_group(pitch_type): Maps pitch types to one of the three groups: Fast, Breaking, or Offspeed.

  prepare_target(trackman): Cleans and prepares the target variable, Pitch_Value:
    - Replaces specific PitchCall and PitchResult values.
    - Merges with the count_transitions.csv dataset to include Pitch_Value.

  prepare_features(trackman, heights):
    - Filters by Level = D1.
    - Adds calculated feature ArmAngle
    - Assigns pitch groups and splits data into three buckets (Fast, Breaking, Offspeed).

  X_y_splits(df_pitchgroup): Splits data into features (X) and target (y) for training and testing.

  train_model(X_train, y_train): Trains an XGBoost model on the input data.

  test_model(model, X_test, y_test, pitchgroup, accuracy_df, model_str): Evaluates the model and appends results (MSE, RMSE, MAE) to the accuracy DataFrame.

  create_score_models(df_pitchgroup, pitchgroup, accuracy_df): Trains, tests, and saves a model for each pitch group. Results are appended to accuracy_df.

**Pipeline:**
1. **Data Loading**:
   - trackman.csv: Raw TrackMan pitch data.
   - heights.csv: Player height and additional information.

2. **Data Preparation**:
   - Clean and merge datasets.
   - Generate additional features.
   - Filter and group data into three pitch types.

3. **Model Training & Testing**:
   - Split data into training and testing sets.
   - Train an XGBoost model for each pitch group.
   - Evaluate model performance.

4. **Model Results**:
   - Save the trained model (xgb_stuff_plus_{pitchgroup}_v{version}.pkl).
   - Output accuracy metrics (accuracy_df).
  
5. **Model Predictions:**
   - Uses the trained model to predict run values for each pitches.
   - overall_grading.py takes team code input and grades each pitch for pitchers on that team. 
