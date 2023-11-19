# FOOD FORECAST : AN INNOVATIVE FOOD PREDICTION SYSTEM


This project is a predictive analytics application designed to forecast food preferences based on historical data. It leverages Bayesian network models and statistical analysis to predict the most likely food choices for each day, considering various factors such as day of the week, the year of study, and past food preferences.

## Features

### Bayesian Network Model
- Incorporates probabilistic dependencies among different food choices.
- Adapts to changing food preferences over time using historical data.
- Enables efficient prediction by modeling the relationships between variables.
### Individual Preferences
- Classifies students into different study years for personalized predictions.
- Considers demographic factors to tailor food recommendations to each student.
### Count Prediction Model
- Utilizes machine learning techniques to predict the count of each food item.
- Accounts for various factors such as historical consumption patterns and student preferences.
- Enhances accuracy in forecasting by addressing individual variations.
### Visualizations
- Presents intuitive charts and graphs illustrating predicted food percentages.
- Enables a comprehensive view of forecasted preferences for effective decision-making.
- Supports data-driven insights for hostel mess management.

## Requirements

- Python 3.x
- Required Python packages: pandas, numpy, matplotlib, scikit-learn, pgmpy

## Architecture Diagram/Flow

![OUTPUT](./output%20img/METHD.png)
![output](./output%20img/ARCH.png)

## Installation

1. Clone the repository:

   ```shell
    git clone https://github.com/your-username/food-forecast.git

2. Install the required packages:

   ```shell
   pip install -r requirements.txt
3. Download the dataset and place it in the project directory.
(hf.csv file for model-1(preference ananlysis) , forbaysian2.csv file for model-2(food-item prediction), data1.csv file for model-3(count of food items))

## Usage

1. Run the forecasting system:
   ```shell
   python food_forecast.py
   ```

2. Access the predicted food preferences for each day.
3. Explore visualizations to understand the forecasted percentages.
4. Install this package before model-2
    ```shell
    !pip install pgmpy
    ```
4. Enter the day to find out what food-items are prefered.
5. prefered food-item count will be displayed. 



## Program:
# VISUALIZING PREFERED FOOD CONSUMPTION
```python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

df = pd.read_csv("Hf.csv")
df.head(10)

df_copy = df.copy()
df_copy = df_copy.drop(columns=['Timestamp','Email Address'])
df_copy.head(10)

df_1 = df_copy.rename(columns={'What are the preferred breakfast choices on Monday?': 'Monday',
                               'What are the preferred breakfast choices on Tuesday?':'Tuesday',
                               'What are the preferred breakfast choices on Wednesday?':'Wednesday',
                               'What are the preferred breakfast choices on Thursday?':'Thursday',
                               'What are the preferred breakfast choices on Friday?':'Friday',
                               'What are the preferred breakfast choices on Saturday?':'Saturday',
                               'What are the preferred breakfast choices on Sunday?':'Sunday'})

df_1['Register Number'] = df_1['Register Number'].str.replace(' ', '')

data_types = df_1.dtypes
print(data_types)
df_1['Register Number'] = df_1['Register Number'].str.strip()

def label_year(row):
    register_number = str(row['Register Number'])
    if register_number.startswith(('2300', '2200', '2301')):
        return 'First Year'
    elif register_number.startswith('212221'):
        return 'Third Year'
    elif register_number.startswith('212222'):
        return 'Second Year'
    elif register_number.startswith('212220'):
        return 'Final Year'
    else:
      return 'NaN'

df_1['Year'] = df_1.apply(label_year, axis=1)

df_1.head(144)

mask = df_1.isna().any(axis=1)
rows_with_nan = df_1[mask]
print(rows_with_nan)

df2 = df_1.dropna()
mask = df2.isna().any(axis=1)
rows_with_nan = df2[mask]
print(rows_with_nan)

df2.tail()

year_counts = df2['Year'].value_counts()

print("Count of First Years:", year_counts.get('First Year', 0))
print("Count of Second Years:", year_counts.get('Second Year', 0))
print("Count of Third Years:", year_counts.get('Third Year', 0))
print("Count of Final Years:", year_counts.get('Final Year', 0))

food_items = ['Idly', 'Dosai']

food_item_percentages = {}

total_counts = df2['Monday'].str.count('|'.join(food_items)).sum()

for food_item in food_items:
    for year in df2['Year'].unique():
        year_df = df2[df2['Year'] == year]
        food_item_count = year_df['Monday'].str.count(food_item).sum()
        percentage = (food_item_count / total_counts) * 100
        key = f"{food_item} - {year}"
        food_item_percentages[key] = percentage

# Print the results
for key, percentage in food_item_percentages.items():
    print(f"Percentage of {key} consumed: {percentage:.2f}%")

import pandas as pd

food_items = {
    'Monday': ['Idly', 'Dosai'],
    'Tuesday': ['Idly', 'Uthappam'],
    'Wednesday': ['Idly','Poori'],
    'Thursday': ['Idly','Veg Kitchdi'],
    'Friday': ['Idly','Pongal'],
    'Saturday': ['Idly','Semiya Upma'],
    'Sunday': ['Egg Noodles','Veg Noodles']
}

food_item_percentages = {}

for day, items in food_items.items():
    for food_item in items:
        food_item_count = df2[day].str.count(food_item).sum()
        percentage = (food_item_count / len(df2)) * 100
        key = f"{food_item} - {day}"
        food_item_percentages[key] = percentage
# for key, percentage in food_item_percentages.items():
#   print(f"{key}: {percentage:.1f}%")
#   print('\n')

    labels = [key for key in food_item_percentages.keys() if day in key]
    sizes = [food_item_percentages[label] for label in labels]
    explode = (0.1,) * len(labels)

    plt.figure(figsize=(4,4))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f"Percentage of Food Consumption on {day}")
    plt.show()
    print('\n')

food_item_percentages = {}

for day, items in food_items.items():
    for food_item in items:
        food_item_count = df2[day].str.count(food_item).sum()
        percentage = (food_item_count / len(df2)) * 100
        key = f"{food_item} - {day}"
        food_item_percentages[key] = percentage

for key, percentage in food_item_percentages.items():
    print(f"{key}: {percentage:.1f}%")
    print('\n')

import pandas as pd

food_items = {
    'Monday': ['Idly', 'Dosai'],
    'Tuesday': ['Idly', 'Uthappam'],
    'Wednesday': ['Idly', 'Poori'],
    'Thursday': ['Idly', 'Veg Kitchdi'],
    'Friday': ['Idly', 'Pongal'],
    'Saturday': ['Idly', 'Semiya Upma'],
    'Sunday': ['Egg Noodles', 'Veg Noodles']
}

food_item_percentages = {}

# Assuming df2 is defined somewhere in your code
# food_item_count = df2[day].str.count(food_item).sum()

for day, items in food_items.items():
    total_count = sum(df2[day].str.count(item).sum() for item in items)

    for food_item in items:
        food_item_count = df2[day].str.count(food_item).sum()
        percentage = (food_item_count / total_count) * 100
        key = f"{food_item} - {day}"
        food_item_percentages[key] = percentage

# Normalize percentages to ensure they add up to 100%
for key, percentage in food_item_percentages.items():
    print(f"{key}: {percentage:.1f}%")

# Verify that the total adds up to 100%
total_percentage = sum(food_item_percentages.values())
print(f"Total Percentage: {total_percentage:.1f}%")
```
# IMPLEMENTING BAYESIAN MODEL TO PREDICT FOOD ITEMS FOR A DAY
```PYTHON
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import LabelEncoder

df_bay = pd.read_csv('forbaysian2.csv')

df_bay

# Identify the correct column names for each food item being chosen
idly_chosen_column = df_bay.columns[3]
dosa_chosen_column = df_bay.columns[4]
pongal_chosen_column = df_bay.columns[5]
uthappam_chosen_column = df_bay.columns[6]
poori_chosen_column = df_bay.columns[7]
veg_kitchidi_chosen_column = df_bay.columns[8]
seviyan_upma_chosen_column = df_bay.columns[9]
egg_noodles_chosen_column = df_bay.columns[10]
veg_noodles_chosen_column = df_bay.columns[11]

pongal_chosen_column

# Assuming 'DAYS' is the column representing days
features = ['DAYS']

# Encode the 'DAYS' column
label_encoder = LabelEncoder()
df_bay['DAYS'] = label_encoder.fit_transform(df_bay['DAYS'])

df_bay['DAYS']

# Create a Bayesian Network model with nodes for 'DAYS' and all food items
model_structure = [
    ('DAYS', idly_chosen_column),
    ('DAYS', dosa_chosen_column),
    ('DAYS', pongal_chosen_column),
    ('DAYS', uthappam_chosen_column),
    ('DAYS', poori_chosen_column),
    ('DAYS', veg_kitchidi_chosen_column),
    ('DAYS', seviyan_upma_chosen_column),
    ('DAYS', egg_noodles_chosen_column),
    ('DAYS', veg_noodles_chosen_column),
   
]
model = BayesianNetwork(model_structure)

# Use MaximumLikelihoodEstimator to estimate CPDs based on the entire dataset
model.fit(df_bay, estimator=MaximumLikelihoodEstimator)

# User input for the day
input_day = input('Enter the day (e.g., MONDAY): ') 
encoded_input_day = label_encoder.transform([input_day])[0]

# Create a DataFrame for inference
inference_data = pd.DataFrame({features[0]: [encoded_input_day]})

# Perform inference for the given day
predicted_data = model.predict(inference_data)

# Filter the predicted data to show only food items with a value of 1

selected_columns = predicted_data.columns[predicted_data.iloc[0] == 1].tolist()

print(selected_columns)
```
## Output:
### FOOD CONSUMPTION 
#### Data Set
![dataset](./output%20img/preferencedataset.png)

#### Preferd food items consumption on a week.
![graph](./output%20img/montus.png)
![graph](./output%20img/wedthurs.png)
![graph](./output%20img/frisat.png)
![graph](./output%20img/sunday.png)
![]()

### FOOD-ITEM PREDICTION 
#### Data Set
![dataset](./output%20img/predictiondataset.png)
#### PREDICTED FOOD FOR USER INPUT DAY
![OUTPUT](./output%20img/input.png)
![OUTPUT](./output%20img/fooditem.png)


## Result:

The Food Forecasting System is an intelligent application that predicts the food preferences for each day based on historical data. By analyzing patterns and considering factors such as the day of the week , the system provides accurate forecasts. The visualizations enhance the interpretability of the predictions, making it a valuable tool for hostel mess management.

This project is useful for optimizing food planning, reducing wastage, and ensuring a better dining experience for students. It can be further extended to incorporate real-time data and additional features for more accurate predictions.


