#!/usr/bin/env python
# coding: utf-8

# # Problem: Predicting Airplane Delays
# 
# The goals of this notebook are:
# - Process and create a dataset from downloaded ZIP files
# - Exploratory data analysis (EDA)
# - Establish a baseline model and improve it
# 
# ## Introduction to business scenario
# You work for a travel booking website that is working to improve the customer experience for flights that were delayed. The company wants to create a feature to let customers know if the flight will be delayed due to weather when the customers are booking the flight to or from the busiest airports for domestic travel in the US. 
# 
# You are tasked with solving part of this problem by leveraging machine learning to identify whether the flight will be delayed due to weather. You have been given access to the a dataset of on-time performance of domestic flights operated by large air carriers. You can use this data to train a machine learning model to predict if the flight is going to be delayed for the busiest airports.
# 
# ### Dataset
# The provided dataset contains scheduled and actual departure and arrival times reported by certified US air carriers that account for at least 1 percent of domestic scheduled passenger revenues. The data was collected by the Office of Airline Information, Bureau of Transportation Statistics (BTS). The dataset contains date, time, origin, destination, airline, distance, and delay status of flights for flights between 2014 and 2018.
# The data are in 60 compressed files, where each file contains a CSV for the flight details in a month for the five years (from 2014 - 2018). The data can be downloaded from this link: [https://ucstaff-my.sharepoint.com/:f:/g/personal/ibrahim_radwan_canberra_edu_au/Er0nVreXmihEmtMz5qC5kVIB81-ugSusExPYdcyQTglfLg?e=bNO312]. Please download the data files and place them on a relative path. Dataset(s) used in this assignment were compiled by the Office of Airline Information, Bureau of Transportation Statistics (BTS), Airline On-Time Performance Data, available with the following link: [https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ]. 

# # Step 1: Problem formulation and data collection
# 
# Start this project off by writing a few sentences below that summarize the business problem and the business goal you're trying to achieve in this scenario. Include a business metric you would like your team to aspire toward. With that information defined, clearly write out the machine learning problem statement. Finally, add a comment or two about the type of machine learning this represents. 
# 
# 
# ### 1. Determine if and why ML is an appropriate solution to deploy.

# Machine learning would be an appropriate solution to deploy in this case as predicting flight delays is a complex problem and their are many factors which impact the flightdelay like historical weather data, airport conditions and so on. ML can identify patterns and predict successfully. ML allows for feature engineering and thus allowing for extractiona and utilisation of relevant features.
# Also, as more data becomes available and algorithms are improved over time, machine learning models can continue to get better. This is in line with the objective of enhancing the client experience by making predictions that are more accurate.

# ### 2. Formulate the business problem, success metrics, and desired ML output.

# The business problem in this case is to improve the customer experience by letting them know about potential delays when booking flights to or from the busiest airports in US.In order to reduce customer dissatisfaction , a machine learning model which can predict the likelihood whether flight will be delayed or not due to weather  will help customers make informed decisions when booking flights.
# 
# Success Metrics: We would measure the accuracy, precision, recall and F1 score of the machine learning model in predicting weather-related flight delays
# 
# The desired ML output is a predictive model that will classify flights - likely to be delayed or not likely to be delayed
# 

# ### 3. Identify the type of ML problem you’re dealing with.

# In this scenario, we are dealing with a supervised binary classification problem. It is supervised as we have a labelled dataset  and can easily create the flag for whether flight delayed or not. 
# The machine learning model will be trained to predict whether a flight will experience a weather-related delay or not, making it a supervised binary classification problem.

# ### Setup
# 
# Now that we have decided where to focus our energy, let's set things up so you can start working on solving the problem.

# In[229]:


import os
from pathlib2 import Path
from zipfile import ZipFile
import time

import pandas as pd
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

# <please add any other library or function you are aiming to import here>


# # Step 2: Data preprocessing and visualization  
# In this data preprocessing phase, you should take the opportunity to explore and visualize your data to better understand it. First, import the necessary libraries and read the data into a Pandas dataframe. After that, explore your data. Look for the shape of the dataset and explore your columns and the types of columns you're working with (numerical, categorical). Consider performing basic statistics on the features to get a sense of feature means and ranges. Take a close look at your target column and determine its distribution.
# 
# ### Specific questions to consider
# 1. What can you deduce from the basic statistics you ran on the features? 
# 
# 2. What can you deduce from the distributions of the target classes?
# 
# 3. Is there anything else you deduced from exploring the data?

# Start by bringing in the dataset from an Amazon S3 public bucket to this notebook environment.

# In[230]:


# download the files

# <note: make them all relative, absolute path is not accepted>
outer_zip_url  = '../OneDrive_1_10-29-2023.zip'
base_path = './'
csv_base_path = '../ExtractedData'



# In[231]:


# How many zip files do we have? write a code to answer it. Have extracted the files as well in this code

import os
import zipfile
import requests


# Create the directory if it doesn't exist
os.makedirs(csv_base_path, exist_ok=True)

# Initialize a counter for zip files
zip_file_count = 0

# Open the main zip file (outer_zip_url)
with zipfile.ZipFile(outer_zip_url, "r") as main_zip:
    # Loop through the files within the main zip file
    for file_info in main_zip.infolist():
        # Extract the file to the specified directory (csv_base_path)
        main_zip.extract(file_info, csv_base_path)

        # Check if the extracted file is a zip file
        if file_info.filename.endswith(".zip"):
            zip_file_count += 1
            # Open the nested zip file and extract its contents
            nested_zip_path = os.path.join(csv_base_path, file_info.filename)
            with zipfile.ZipFile(nested_zip_path, "r") as nested_zip:
                # Extract the contents of the nested zip file to the same directory (csv_base_path)
                nested_zip.extractall(csv_base_path)

print(f"Extraction completed. Found {zip_file_count} zip files.")


# #### Extract CSV files from ZIP files

# In[232]:


# How many csv files have we extracted? write a code to answer it.
csv_files = [filename for filename in os.listdir(csv_base_path) if filename.endswith(".csv")]

# Count the number of CSV files
csv_file_count = len(csv_files)

print(f"Number of CSV files extracted: {csv_file_count}")


# Before loading the CSV file, read the HTML file from the extracted folder. This HTML file includes the background and more information on the features included in the dataset.

# In[233]:


from IPython.display import IFrame

IFrame(src=os.path.relpath(f"{csv_base_path}/readme.html"), width=1000, height=600)


# #### Load sample CSV
# 
# Before combining all the CSV files, get a sense of the data from a single CSV file. Using Pandas, read the `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2018_9.csv` file first. You can use the Python built-in `read_csv` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)).

# In[234]:


import os

# Define the specific file name
file_name = 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2018_9.csv'  

# Construct the full file path by joining the base path and file name
full_file_path = os.path.join(csv_base_path, file_name)

# Read the CSV file into a Pandas DataFrame
import pandas as pd
df = pd.read_csv(full_file_path)

# Display the first few rows of the DataFrame to get a sense of the data
print(df.head())


# **Question**: Print the row and column length in the dataset, and print the column names.

# In[235]:


num_rows, num_columns = df.shape

# Display the number of rows and columns
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Display the column names
print("Column names:")
print(df.columns)


# **Question**: Print the first 10 rows of the dataset.  

# In[236]:


# Enter your code here
print(df.head(10))


# **Question**: Print all the columns in the dataset. Use `<dataframe>.columns` to view the column names.

# In[237]:


print(f'The column names are:')
print('#########')
for col in df.columns:
    print(col)


# **Question**: Print all the columns in the dataset that contain the word 'Del'. This will help you see how many columns have delay data in them.
# 
# **Hint**: You can use a Python list comprehension to include values that pass certain `if` statement criteria.
# 
# For example: `[x for x in [1,2,3,4,5] if x > 2]`  
# 
# **Hint**: You can use the `in` keyword ([documentation](https://www.w3schools.com/python/ref_keyword_in.asp)) to check if the value is in a list or not. 
# 
# For example: `5 in [1,2,3,4,5]`

# In[238]:


# Enter your code here
delay_columns = [col for col in df.columns if 'Del' in col]

print("Columns with 'Del' in their names:")
for col in delay_columns:
    print(col)


# Here are some more questions to help you find out more about your dataset.
# 
# **Questions**   
# 1. How many rows and columns does the dataset have?   
# 2. How many years are included in the dataset?   
# 3. What is the date range for the dataset?   
# 4. Which airlines are included in the dataset?   
# 5. Which origin and destination airports are covered?

# In[239]:


# to answer above questions, complete the following code
#print("The years in this dataset are: ", <CODE>)
#print("The months covered in this dataset are: ", <CODE>)
#print("The date range for data is :" , min(<CODE>), " to ", max(<CODE>))
#print("The airlines covered in this dataset are: ", list(<CODE>))
#print("The Origin airports covered are: ", list(<CODE>))
#print("The Destination airports covered are: ", list(<CODE>))

# Have added the code in below cells


# In[240]:


# How many rows and columns does the dataset have?
num_rows, num_columns = df.shape
print("The #rows and #columns are", num_rows, "and", num_columns)


# In[241]:


# How many years are included in the dataset?
years = df['Year'].unique()
print("The years in this dataset are:", list(years))


# In[242]:


# Months covered in the dataset
import calendar
months = df['Month'].unique()
month_names = [calendar.month_name[month] for month in range(1, 13) if month in months]
print("The months covered in this dataset are: ", month_names)


# In[243]:


# What is the date range for the dataset?

min_date = df[['Year', 'Month' ,'DayofMonth']].min().values
max_date = df[['Year', 'Month' , 'DayofMonth']].max().values



# Create a string representation of the date range
min_date_str = f"{int(min_date[1]):02d}/{int(min_date[2]):02d}/{int(min_date[0])}"
max_date_str = f"{int(max_date[1]):02d}/{int(max_date[2]):02d}/{int(max_date[0])}"
date_range_str = f"{min_date_str} to {max_date_str}"

print("The date range for data is:", date_range_str)



# In[244]:


airlines = df['Reporting_Airline'].unique()
print("The airlines covered in this dataset are:", list(airlines))


# In[245]:


# Which origin and destination airports are covered?
origin_airports = df['OriginCityName'].unique()
destination_airports = df['DestCityName'].unique()
print("The Origin airports covered are:", list(origin_airports))
print("The Destination airports covered are:", list(destination_airports))


# **Question**: What is the count of all the origin and destination airports?
# 
# **Hint**: You can use the Pandas `values_count` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)) to find out the values for each airport using the columns `Origin` and `Dest`.

# In[246]:


import pandas as pd

# Count of unique origin airports
origin_airport_count = df['OriginAirportID'].nunique()

# Count of unique destination airports
destination_airport_count = df['DestAirportID'].nunique()

# Create a DataFrame with scalar values and specify the index
counts = pd.Series({'Origin': origin_airport_count, 'Destination': destination_airport_count})

# Convert the Series to a DataFrame
counts_df = counts.to_frame().T

# Display the DataFrame
print(counts_df)


# **Question**: Print the top 15 origin and destination airports based on number of flights in the dataset.
# 
# **Hint**: You can use the Pandas `sort_values` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)).

# In[247]:


#counts.sort_values(by=<CODE>,ascending=False).head(15 )# Enter your code here

origin_counts = df['Origin'].value_counts()
destination_counts = df['Dest'].value_counts()

# Print the top 15 origin and destination airports based on the number of flights
top_15_origin = origin_counts.sort_values(ascending=False).head(15)
top_15_destination = destination_counts.sort_values(ascending=False).head(15)

print("Top 15 origin airports based on number of flights:")
print(top_15_origin)

print("\nTop 15 destination airports based on number of flights:")
print(top_15_destination)


# **Question**: Given all the information about a flight trip, can you predict if it would be delayed?

# With the given information  about fligh trip, we can predict if the flight will be delayed. The factors such as date, time, origin, destination , airline, delay information can help with the prediction. The machine learning model can be trained using this data of past flight data that has labels designating whether or not a flight was delayed. This allows the model to identify patterns and relationships in the data that can be used to forecast the likelihood of a delay in a new flight.

# Now, assume you are traveling from San Francisco to Los Angeles on a work trip. You want to have an ideas if your flight will be delayed, given a set of features, so that you can manage your reservations in Los Angeles better. How many features from this dataset would you know before your flight?
# 
# Columns such as `DepDelay`, `ArrDelay`, `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`, and `DivArrDelay` contain information about a delay. But this delay could have occured at the origin or destination. If there were a sudden weather delay 10 minutes before landing, this data would not be helpful in managing your Los Angeles reservations.
# 
# So to simplify the problem statement, consider the following columns to predict an arrival delay:<br>
# 
# `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`, `FlightDate`, `Reporting_Airline`, `Origin`, `OriginState`, `Dest`, `DestState`, `CRSDepTime`, `DepDelayMinutes`, `DepartureDelayGroups`, `Cancelled`, `Diverted`, `Distance`, `DistanceGroup`, `ArrDelay`, `ArrDelayMinutes`, `ArrDel15`, `AirTime`
# 
# You will also filter the source and destination airports to be:
# - Top airports: ATL, ORD, DFW, DEN, CLT, LAX, IAH, PHX, SFO
# - Top 5 airlines: UA, OO, WN, AA, DL
# 
# This should help in reducing the size of data across the CSV files to be combined.

# #### Combine all CSV files
# 
# **Hint**:  
# First, create an empy dataframe that you will use to copy your individual dataframes from each file. Then, for each file in the `csv_files` list:
# 
# 1. Read the CSV file into a dataframe  
# 2. Filter the columns based on the `filter_cols` variable
# 
# ```
#         columns = ['col1', 'col2']
#         df_filter = df[columns]
# ```
# 
# 3. Keep only the subset_vals in each of the subset_cols. Use the `isin` Pandas function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html)) to check if the `val` is in the dataframe column and then choose the rows that include it.
# 
# ```
#         df_eg[df_eg['col1'].isin('5')]
# ```
# 
# 4. Concatenate the dataframe with the empty dataframe 

# In[248]:


def combine_csv(csv_files, filter_cols, subset_cols, subset_vals, file_name):
    """
    Combine csv files into one Data Frame
    csv_files: list of csv file paths
    filter_cols: list of columns to filter
    subset_cols: list of columns to subset rows
    subset_vals: list of list of values to subset rows
    """
    # Create an empty dataframe
    df1 =  pd.DataFrame(columns=filter_cols)# Enter your code here 
    
    
    # Loop through the list of CSV files
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        current_df = pd.read_csv(csv_file)

        # Filter the columns
        current_df = current_df[filter_cols]

        # Filter the rows based on subset columns and values
        for col, vals in zip(subset_cols, subset_vals):
            current_df = current_df[current_df[col].isin(vals)]

        # Concatenate the current DataFrame with the main DataFrame
        df1 = pd.concat([df1, current_df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    df1.to_csv(file_name, index=False)
    
    #<complete the code of this function>


# In[249]:


#cols is the list of columns to predict Arrival Delay 
cols = ['Year','Quarter','Month','DayofMonth','DayOfWeek','FlightDate',
        'Reporting_Airline','Origin','OriginState','Dest','DestState',
        'CRSDepTime','Cancelled','Diverted','Distance','DistanceGroup',
        'ArrDelay','ArrDelayMinutes','ArrDel15','AirTime']

subset_cols = ['Origin', 'Dest', 'Reporting_Airline']

# subset_vals is a list collection of the top origin and destination airports and top 5 airlines
subset_vals = [['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['UA', 'OO', 'WN', 'AA', 'DL']]


# In[250]:


# Use os.listdir to list all files in the directory
all_files = os.listdir(csv_base_path)

# Filter the list to include only CSV files (ending with .csv)
csv_files = [os.path.join(csv_base_path, filename) for filename in all_files if filename.endswith(".csv")]


# Use the function above to merge all the different files into a single file that you can read easily. 
# 
# **Note**: This will take 5-7 minutes to complete.

# In[251]:


combined_csv_filename = f"{base_path}combined_flight_data.csv"


# In[252]:


start = time.time()

combined_csv_filename = f"{base_path}combined_flight_data.csv"

# < write code to call the combined_csv function>
combine_csv(csv_files, cols, subset_cols, subset_vals, 'combined_flight_data.csv')
print(f'csv\'s merged in {round((time.time() - start)/60,2)} minutes')


# #### Load dataset
# 
# Load the combined dataset.

# In[253]:


data = pd.read_csv(combined_csv_filename)# Enter your code here to read the combined csv file.


# Print the first 5 records.

# In[254]:


# Enter your code here 
print(data.head())


# Here are some more questions to help you find out more about your dataset.
# 
# **Questions**   
# 1. How many rows and columns does the dataset have?   
# 2. How many years are included in the dataset?   
# 3. What is the date range for the dataset?   
# 4. Which airlines are included in the dataset?   
# 5. Which origin and destination airports are covered?

# In[255]:


# to answer above questions, complete the following code
#print("The #rows and #columns are ", <CODE> , " and ", <CODE>)
#print("The years in this dataset are: ", list(<CODE>))
#print("The months covered in this dataset are: ", sorted(list(<CODE>)))
#print("The date range for data is :" , min(<CODE>), " to ", max(<CODE>))
#print("The airlines covered in this dataset are: ", list(<CODE>))
#print("The Origin airports covered are: ", list(<CODE>))
#print("The Destination airports covered are: ", list(<CODE>))


# these questions have been answerd in below cells


# In[256]:


num_rows, num_columns = data.shape
print("The #rows and #columns are", num_rows, "and", num_columns)


# In[257]:


years = data['Year'].unique()
print("The years in this dataset are:", list(years))


# In[258]:


# Months covered in the dataset
import calendar
months = data['Month'].unique()
month_names = [calendar.month_name[month] for month in range(1, 13) if month in months]
print("The months covered in this dataset are: ", month_names)


# In[259]:


# What is the date range for the dataset?

min_date = data[['Year', 'Month' ,'DayofMonth']].min().values
max_date = data[['Year', 'Month' , 'DayofMonth']].max().values

# Create a string representation of the date range
min_date_str = f"{int(min_date[1]):02d}/{int(min_date[2]):02d}/{int(min_date[0])}"
max_date_str = f"{int(max_date[1]):02d}/{int(max_date[2]):02d}/{int(max_date[0])}"
date_range_str = f"{min_date_str} to {max_date_str}"

print("The date range for data is:", date_range_str)


# In[260]:


airlines = data['Reporting_Airline'].unique()
print("The airlines covered in this dataset are:", list(airlines))


# In[261]:


# Which origin and destination airports are covered?
origin_airports = data['Origin'].unique()
destination_airports = data['Dest'].unique()
print("The Origin airports covered are:", list(origin_airports))
print("The Destination airports covered are:", list(destination_airports))


# Let's define our **target column : is_delay** (1 - if arrival time delayed more than 15 minutes, 0 - otherwise). Use the `rename` method to rename the column from `ArrDel15` to `is_delay`.
# 
# **Hint**: You can use the Pandas `rename` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html)).
# 
# For example:
# ```
# df.rename(columns={'col1':'column1'}, inplace=True)
# ```

# In[262]:


# Rename the 'ArrDel15' column to 'is_delay'
data.rename(columns={'ArrDel15': 'is_delay'}, inplace=True)


# Look for nulls across columns. You can use the `isnull()` function ([documentation](https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.isnull.html)).
# 
# **Hint**: `isnull()` detects whether the particular value is null or not and gives you a boolean (True or False) in its place. Use the `sum(axis=0)` function to sum up the number of columns.

# In[263]:


# Enter your code here
# Check for null values in each column
null_counts = data.isnull().sum(axis=0)

# Print the number of null values in each column
print("Null counts across columns:")
print(null_counts)


# The arrival delay details and airtime are missing for 22540 out of 1658130 rows, which is 1.3%. You can either remove or impute these rows. The documentation does not mention anything about missing rows.
# 
# **Hint**: Use the `~` operator to choose the values that aren't null from the `isnull()` output.
# 
# For example:
# ```
# null_eg = df_eg[~df_eg['column_name'].isnull()]
# ```

# In[264]:


### Remove null columns
#data = # Enter your code here

# Select rows without missing values in the specified columns
data = data[~data['ArrDelay'].isnull() & ~data['ArrDelayMinutes'].isnull() & ~data['is_delay'].isnull() & ~data['AirTime'].isnull()]

# Reset the index after removing rows
data.reset_index(drop=True, inplace=True)


# Get the hour of the day in 24-hour time format from CRSDepTime.

# In[265]:


# Extract the hour of the day from 'CRSDepTime' as an integer
data['DepHourofDay'] = data['CRSDepTime'] // 100

# Reset the column data type to integer (optional, if it's not already integer)
data['DepHourofDay'] = data['DepHourofDay'].astype(int)


# ## **The ML problem statement**
# - Given a set of features, can you predict if a flight is going to be delayed more than 15 minutes?
# - Because the target variable takes only 0/1 value, you could use a classification algorithm. 

# ### Data exploration
# 
# #### Check class delay vs. no delay
# 
# **Hint**: Use a `groupby` plot ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)) with a `bar` plot ([documentation](https://matplotlib.org/tutorials/introductory/pyplot.html)) to plot the frequency vs. distribution of the class.

# In[266]:


# Group the data by the 'is_delay' column and calculate the frequency of each class
class_distribution = (data.groupby('is_delay').size() / len(data)).plot(kind='bar')

# Add labels and title to the plot
plt.ylabel('Frequency')
plt.title('Distribution of classes')

# Show the plot
plt.show()


# **Question**: What can you deduce from the bar plot about the ratio of delay vs. no delay?

# 1. The bars in the plot represent the frequency of each class. Here , we can see that no_delay(represented by 0) occurs more frequently than likely delay.
# 2. It is an imbalanced dataset as one class (likely 'no delay') appears to be more frequent than the other.
# 

# **Questions**: 
# 
# - Which months have the most delays?
# - What time of the day has the most delays?
# - What day of the week has the most delays?
# - Which airline has the most delays?
# - Which origin and destination airports have the most delays?
# - Is flight distance a factor in the delays?

# In[267]:


viz_columns = ['Month', 'DepHourofDay', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest']
fig, axes = plt.subplots(3, 2, figsize=(20,20), squeeze=False)
# fig.autofmt_xdate(rotation=90)

for idx, column in enumerate(viz_columns):
    ax = axes[idx//2, idx%2]
    temp = data.groupby(column)['is_delay'].value_counts(normalize=True).rename('percentage').\
    mul(100).reset_index().sort_values(column)
    sns.barplot(x=column, y="percentage", hue="is_delay", data=temp, ax=ax)
    plt.ylabel('% delay/no-delay')
    

plt.show()


# In[268]:


sns.lmplot( x="is_delay", y="Distance", data=data, fit_reg=False, hue='is_delay', legend=False)
plt.legend(loc='center')
plt.xlabel('is_delay')
plt.ylabel('Distance')
plt.show()


# 1. Which months have the most delays?
# 
# Based on the 'Month' subplot, it appears that certain months, such as June, July, and August, have a higher percentage of delays compared to other months. 
# 
# 2. What time of the day has the most delays?
# The 'DepHourofDay' subplot suggests that flights departing in the evening and night peaking at around 2000 hrs tend to have a higher percentage of delays. These are typically periods with lower visibility and potential weather-related challenges.
# 
# 3. What day of the week has the most delays?
# The 'DayOfWeek' subplot shows that flights on Saturdays and Sundays tend to have a higher percentage of delays compared to other weekdays. 
# 
# 4. Which airline has the most delays?
# The 'Reporting_Airline' subplot displays the percentage of delays for each airline. It appears that DL  airline have a higher percentage of delays compared to others.
# 
# 5. Which origin and destination airports have the most delays?
# The 'Origin' and 'Dest' subplots provide information about the airports associated with the percentage of delays.
# ORD in the origin airports and SFO in the destination airports have the highest number of delays.
# 
# 6. Is flight distance a factor in the delays?
# From the plot, we can infer that distance alone is not a strong predictor. Whether longer or shorter flights are regularly linked to delays is not evident from the data. Flights with delays are scattered over the spectrum of distances, and flights with delays can occur on both short and long flights.

# ### Features
# 
# Look at all the columns and what their specific types are.

# In[269]:


data.columns


# In[270]:


data.dtypes


# Filtering the required columns:
# - Date is redundant, because you have Year, Quarter, Month, DayofMonth, and DayOfWeek to describe the date.
# - Use Origin and Dest codes instead of OriginState and DestState.
# - Because you are just classifying whether the flight is delayed or not, you don't need TotalDelayMinutes, DepDelayMinutes, and ArrDelayMinutes.
# 
# Treat DepHourofDay as a categorical variable because it doesn't have any quantitative relation with the target.
# - If you had to do a one-hot encoding of it, it would result in 23 more columns.
# - Other alternatives to handling categorical variables include hash encoding, regularized mean encoding, and bucketizing the values, among others.
# - Just split into buckets here.
# 
# **Hint**: To change a column type to category, use the `astype` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html)).

# In[271]:


data_orig = data.copy()
data = data[[ 'is_delay', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest','Distance','DepHourofDay']]
categorical_columns  = ['Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest', 'DepHourofDay']
for c in categorical_columns:
    data[c] = data[c].astype('category')


# To use one-hot encoding, use the Pandas `get_dummies` function for the categorical columns that you selected above. Then, you can concatenate those generated features to your original dataset using the Pandas `concat` function. For encoding categorical variables, you can also use *dummy encoding* by using a keyword `drop_first=True`. For more information on dummy encoding, see https://en.wikiversity.org/wiki/Dummy_variable_(statistics).
# 
# For example:
# ```
# pd.get_dummies(df[['column1','columns2']], drop_first=True)
# ```

# In[272]:


data_dummies = pd.get_dummies(data[categorical_columns], drop_first=True)
data = pd.concat([data, data_dummies], axis = 1)
data.drop(categorical_columns,axis=1, inplace=True)


# Check the length of the dataset and the new columnms.

# In[273]:


# Enter your code here
# Check the length of the dataset
dataset_length = len(data)

# Check the number of columns in the dataset
num_columns = data.shape[1]

# Print the results
print("Length of the dataset:", dataset_length)
print("Number of columns in the dataset:", num_columns)


# In[274]:


# Enter your code here
data.columns


# **Sample Answer:** 
# ```
# Index(['Distance', 'is_delay', 'Quarter_2', 'Quarter_3', 'Quarter_4',
#        'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
#        'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
#        'DayofMonth_2', 'DayofMonth_3', 'DayofMonth_4', 'DayofMonth_5',
#        'DayofMonth_6', 'DayofMonth_7', 'DayofMonth_8', 'DayofMonth_9',
#        'DayofMonth_10', 'DayofMonth_11', 'DayofMonth_12', 'DayofMonth_13',
#        'DayofMonth_14', 'DayofMonth_15', 'DayofMonth_16', 'DayofMonth_17',
#        'DayofMonth_18', 'DayofMonth_19', 'DayofMonth_20', 'DayofMonth_21',
#        'DayofMonth_22', 'DayofMonth_23', 'DayofMonth_24', 'DayofMonth_25',
#        'DayofMonth_26', 'DayofMonth_27', 'DayofMonth_28', 'DayofMonth_29',
#        'DayofMonth_30', 'DayofMonth_31', 'DayOfWeek_2', 'DayOfWeek_3',
#        'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
#        'Reporting_Airline_DL', 'Reporting_Airline_OO', 'Reporting_Airline_UA',
#        'Reporting_Airline_WN', 'Origin_CLT', 'Origin_DEN', 'Origin_DFW',
#        'Origin_IAH', 'Origin_LAX', 'Origin_ORD', 'Origin_PHX', 'Origin_SFO',
#        'Dest_CLT', 'Dest_DEN', 'Dest_DFW', 'Dest_IAH', 'Dest_LAX', 'Dest_ORD',
#        'Dest_PHX', 'Dest_SFO'],
#       dtype='object')
# ```

# Now you are ready to do model training. Before splitting the data, rename the column `is_delay` to `target`.
# 
# **Hint**: You can use the Pandas `rename` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html)).

# In[275]:


data.rename(columns = {'is_delay':'target'}, inplace=True )# Enter your code here


# In[276]:


# write code to Save the combined csv file (combined_csv_v1.csv) to your local computer
# note this combined file will be used in part B

data.to_csv('combined_csv_v1.csv', index=False)


# # Step 3: Model training and evaluation
# 
# 1. Split the data into `train_data`, and `test_data` using `sklearn.model_selection.train_test_split`.  
# 2. Build a logistic regression model for the data, where training data is 80%, and test data is 20%.
# 
# Use the following cells to complete these steps. Insert and delete cells where needed.
# 

# ### Train test split

# In[277]:


# write Code here to split data into train, validate and test
from sklearn.model_selection import train_test_split


# Define the feature columns (X) and the target column (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ### Baseline classification model

# In[279]:


# <write code here>
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)


# ## Model evaluation
# In this section, you'll evaluate your trained model on test data and report on the following metrics:
# * Confusion Matrix plot
# * Plot the ROC
# * Report statistics such as Accuracy, Percision, Recall, Sensitivity and Specificity

# To view a plot of the confusion matrix, and various scoring metrics, create a couple of functions:

# In[280]:


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(test_labels, target_predicted):
    # complete the code here
    cm = confusion_matrix(test_labels, target_predicted)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['No Delay', 'Delay'], rotation=45)
    plt.yticks(tick_marks, ['No Delay', 'Delay'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    


# In[281]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def plot_roc(test_labels, target_predicted):
    # complete the code here
    fpr, tpr, thresholds = roc_curve(test_labels, target_predicted)
    auc = roc_auc_score(test_labels, target_predicted)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


# To plot the confusion matrix, call the `plot_confusion_matrix` function on the `test_labels` and `target_predicted` data from your batch job:

# In[282]:


# Enter you code here
plot_confusion_matrix(y_test, y_pred)


# To print statistics and plot an ROC curve, call the `plot_roc` function on the `test_labels` and `target_predicted` data from your batch job:

# In[283]:


# Enter you code here

# Plot the ROC curve
plot_roc(y_test, y_pred)

# Report various statistics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ### Key questions to consider:
# 1. How does your model's performance on the test set compare to the training set? What can you deduce from this comparison? 
# 
# 2. Are there obvious differences between the outcomes of metrics like accuracy, precision, and recall? If so, why might you be seeing those differences? 
# 
# 3. Is the outcome for the metric(s) you consider most important sufficient for what you need from a business standpoint? If not, what are some things you might change in your next iteration (in the feature engineering section, which is coming up next)? 
# 
# Use the cells below to answer these and other questions. Insert and delete cells where needed.

# 
# **Question**: What can you summarize from the confusion matrix?
# 

# 1. How does your model's performance on the test set compare to the training set? What can you deduce from this comparison? 
# Accuracy on the test set is approximately 0.7901, which is the proportion of correctly classified instances.
# 2. Are there obvious differences between the outcomes of metrics like accuracy, precision, and recall? If so, why might you be seeing those differences? 
# 
# Precision for class 0 (not delayed) is 0.79, while precision for class 1 (delayed) is 0.56. The lower precision for class 1 indicates that the model may have more false positives among its predictions for delays.
# 
# Recall for class 0 is 1.00, while recall for class 1 is only 0.00.The low recall for class 1 suggests that the model is missing a significant number of actual delay cases.
# 
# F1-score is a combination of precision and recall, and for class 1, it is only 0.01, indicating that the model's ability to correctly classify delayed flights is quite low.
# 
# 3. Is the outcome for the metric(s) you consider most important sufficient for what you need from a business standpoint? If not, what are some things you might change in your next iteration (in the feature engineering section, which is coming up next)? 
# 
# From a business standpoint,the model's recall performance for class 1 (delayed flights) is insufficient. Customers may not receive enough information about possible delays if a model with a low recall is missing a significant number of real delayed flights.
# 
# 4. What can you summarize from the confusion matrix?
# 
# True Positives (TP): 3359 delayed flights were correctly predicted as delayed.
# True Negatives (TN): 213322 non-delayed flights were correctly predicted as non-delayed.
# False Positives (FP): 2513 non-delayed flights were incorrectly predicted as delayed.
# False Negatives (FN): 55160 delayed flights were incorrectly predicted as non-delayed.
# 
# There are a large number of false negatives which is a matter of concern.
# 
# 

# # Step 4: Deployment
# 
# 1. In this step you are required to push your source code and requirements file to a GitLab repository without the data files. Please use the Git commands to complete this task
# 2- Create a “readme.md” markdown file that describes the code of this repository and how to run it and what the user would expect if got the code running.
# 
# In the cell below provide the link of the pushed repository on your GitLab account.
# 

# In[284]:


### Provide a link for your Gitlab repository here


# # Iteration II

# # Step 5: Feature engineering
# 
# You've now gone through one iteration of training and evaluating your model. Given that the outcome you reached for your model the first time probably wasn't sufficient for solving your business problem, what are some things you could change about your data to possibly improve model performance?
# 
# ### Key questions to consider:
# 1. How might the balance of your two main classes (delay and no delay) impact model performance?
# 2. Do you have any features that are correlated?
# 3. Are there feature reduction techniques you could perform at this stage that might have a positive impact on model performance? 
# 4. Can you think of adding some more data/datasets?
# 4. After performing some feature engineering, how does your model performance compare to the first iteration?
# 
# Use the cells below to perform specific feature engineering techniques (per the questions above) that you think could improve your model performance. Insert and delete cells where needed.
# 
# 
# Before you start, think about why the precision and recall are around 80% while the accuracy is 99%.

# #### Add more features
# 
# 1. Holidays
# 2. Weather

# Because the list of holidays from 2014 to 2018 is known, you can create an indicator variable **is_holiday** to mark these.
# The hypothesis is that airplane delays could be higher during holidays compared to the rest of the days. Add a boolean variable `is_holiday` that includes the holidays for the years 2014-2018.

# In[285]:


# Source: http://www.calendarpedia.com/holidays/federal-holidays-2014.html

holidays_14 = ['2014-01-01',  '2014-01-20', '2014-02-17', '2014-05-26', '2014-07-04', '2014-09-01', '2014-10-13', '2014-11-11', '2014-11-27', '2014-12-25' ] 
holidays_15 = ['2015-01-01',  '2015-01-19', '2015-02-16', '2015-05-25', '2015-06-03', '2015-07-04', '2015-09-07', '2015-10-12', '2015-11-11', '2015-11-26', '2015-12-25'] 
holidays_16 = ['2016-01-01',  '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24', '2016-12-25', '2016-12-26']
holidays_17 = ['2017-01-02', '2017-01-16', '2017-02-20', '2017-05-29' , '2017-07-04', '2017-09-04' ,'2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25']
holidays_18 = ['2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28' , '2018-07-04', '2018-09-03' ,'2018-10-08', '2018-11-12','2018-11-22', '2018-12-25']
holidays = holidays_14+ holidays_15+ holidays_16 + holidays_17+ holidays_18

### Add indicator variable for holidays
data_orig['is_holiday'] = data_orig['FlightDate'].apply(lambda x: 1 if x in holidays else 0)# Enter your code here 


# Weather data was fetched from https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00023174,USW00012960,USW00003017,USW00094846,USW00013874,USW00023234,USW00003927,USW00023183,USW00013881&dataTypes=AWND,PRCP,SNOW,SNWD,TAVG,TMIN,TMAX&startDate=2014-01-01&endDate=2018-12-31.
# <br>
# 
# This dataset has information on wind speed, precipitation, snow, and temperature for cities by their airport codes.
# 
# **Question**: Could bad weather due to rains, heavy winds, or snow lead to airplane delay? Let's check!

# In[286]:


# download data from the link above and place it into the data folder


# Import weather data prepared for the airport codes in our dataset. Use the stations and airports below for the analysis, and create a new column called `airport` that maps the weather station to the airport name.

# In[287]:


weather = pd.read_csv("daily-summaries.csv") # Enter your code here to read 'daily-summaries.csv' file
station = ['USW00023174','USW00012960','USW00003017','USW00094846',
           'USW00013874','USW00023234','USW00003927','USW00023183','USW00013881'] 
airports = ['LAX', 'IAH', 'DEN', 'ORD', 'ATL', 'SFO', 'DFW', 'PHX', 'CLT']

### Map weather stations to airport code
station_map = dict(zip(station, airports))# Enter your code here 
weather['airport'] = weather['STATION'].map(station_map)# Enter your code here 


# Create another column called `MONTH` from the `DATE` column.

# In[288]:


weather['MONTH'] = weather['DATE'].apply(lambda x: x.split('-')[1])# Enter your code here 
weather.head()


# ### Sample output
# ```
#   STATION     DATE      AWND PRCP SNOW SNWD TAVG TMAX  TMIN airport MONTH
# 0 USW00023174 2014-01-01 16   0   NaN  NaN 131.0 178.0 78.0  LAX    01
# 1 USW00023174 2014-01-02 22   0   NaN  NaN 159.0 256.0 100.0 LAX    01
# 2 USW00023174 2014-01-03 17   0   NaN  NaN 140.0 178.0 83.0  LAX    01
# 3 USW00023174 2014-01-04 18   0   NaN  NaN 136.0 183.0 100.0 LAX    01
# 4 USW00023174 2014-01-05 18   0   NaN  NaN 151.0 244.0 83.0  LAX    01
# ```

# Analyze and handle the `SNOW` and `SNWD` columns for missing values using `fillna()`. Use the `isna()` function to check the missing values for all the columns.

# In[289]:


weather.SNOW.fillna(0, inplace=True)# Enter your code here
weather.SNWD.fillna(0, inplace=True)# Enter your code here
weather.isna().sum()


# In[290]:


weather


# **Question**: Print the index of the rows that have missing values for TAVG, TMAX, TMIN.
# 
# **Hint**: Use the `isna()` function to find the rows that are missing, and then use the list on the idx variable to get the index.

# In[291]:


idx = np.array([i for i in range(len(weather))])
TAVG_idx = idx[weather['TAVG'].isna()]# Enter your code here 
TMAX_idx = idx[weather['TMAX'].isna()]# Enter your code here 
TMIN_idx = idx[weather['TMIN'].isna()]# Enter your code here 
TAVG_idx,TMAX_idx,TMIN_idx


# ### Sample output
# 
# ```
# array([ 3956,  3957,  3958,  3959,  3960,  3961,  3962,  3963,  3964,
#         3965,  3966,  3967,  3968,  3969,  3970,  3971,  3972,  3973,
#         3974,  3975,  3976,  3977,  3978,  3979,  3980,  3981,  3982,
#         3983,  3984,  3985,  4017,  4018,  4019,  4020,  4021,  4022,
#         4023,  4024,  4025,  4026,  4027,  4028,  4029,  4030,  4031,
#         4032,  4033,  4034,  4035,  4036,  4037,  4038,  4039,  4040,
#         4041,  4042,  4043,  4044,  4045,  4046,  4047, 13420])
# ```

# You can replace the missing TAVG, TMAX, and TMIN with the average value for a particular station/airport. Because the consecutive rows of TAVG_idx are missing, replacing with a previous value would not be possible. Instead, replace it with the mean. Use the `groupby` function to aggregate the variables with a mean value.

# In[292]:


weather_impute = weather.groupby(['MONTH','STATION']).agg({'TAVG':'mean','TMAX':'mean', 'TMIN':'mean' }).reset_index()# Enter your code here
weather_impute.head(2)


# Merge the mean data with the weather data.

# In[293]:


weather_impute.columns


# In[294]:


### get the yesterday's data
weather = pd.merge(weather, weather_impute,  how='left', left_on=['MONTH','STATION'], right_on = ['MONTH','STATION'])\
.rename(columns = {'TAVG_y':'TAVG_AVG',
                   'TMAX_y':'TMAX_AVG', 
                   'TMIN_y':'TMIN_AVG',
                   'TAVG_x':'TAVG',
                   'TMAX_x':'TMAX', 
                   'TMIN_x':'TMIN'})


# Check for missing values again.

# In[295]:


weather.TAVG[TAVG_idx] = weather.TAVG_AVG[TAVG_idx]
weather.TMAX[TMAX_idx] = weather.TMAX_AVG[TMAX_idx]
weather.TMIN[TMIN_idx] = weather.TMIN_AVG[TMIN_idx]
weather.isna().sum()


# Drop `STATION,MONTH,TAVG_AVG,TMAX_AVG,TMIN_AVG,TMAX,TMIN,SNWD` from the dataset

# In[296]:


weather.drop(columns=['STATION','MONTH','TAVG_AVG', 'TMAX_AVG', 'TMIN_AVG', 'TMAX' ,'TMIN', 'SNWD'],inplace=True)


# Add the origin and destination weather conditions to the dataset.

# In[297]:


### Add origin weather conditions
data_orig = pd.merge(data_orig, weather,  how='left', left_on=['FlightDate','Origin'], right_on = ['DATE','airport'])\
.rename(columns = {'AWND':'AWND_O','PRCP':'PRCP_O', 'TAVG':'TAVG_O', 'SNOW': 'SNOW_O'})\
.drop(columns=['DATE','airport'])

### Add destination weather conditions
data_orig = pd.merge(data_orig, weather,  how='left', left_on=['FlightDate','Dest'], right_on = ['DATE','airport'])\
.rename(columns = {'AWND':'AWND_D','PRCP':'PRCP_D', 'TAVG':'TAVG_D', 'SNOW': 'SNOW_D'})\
.drop(columns=['DATE','airport'])


# **Note**: It is always a good practice to check nulls/NAs after joins.

# In[298]:


sum(data_orig.isna().any())


# In[299]:


data_orig.dropna(inplace=True)


# In[300]:


data_orig.columns


# Convert the categorical data into numerical data using one-hot encoding.

# In[301]:


data = data_orig.copy()
data = data[['is_delay', 'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest','Distance','DepHourofDay','is_holiday', 'AWND_O', 'PRCP_O',
       'TAVG_O', 'AWND_D', 'PRCP_D', 'TAVG_D', 'SNOW_O', 'SNOW_D']]


categorical_columns  = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest', 'is_holiday']
for c in categorical_columns:
    data[c] = data[c].astype('category')


# In[302]:


data_dummies = pd.get_dummies(data[categorical_columns], drop_first=True) # Enter your code here
data = pd.concat([data, data_dummies], axis = 1)
data.drop(categorical_columns,axis=1, inplace=True)


# ### Sample code
# 
# ```
# data_dummies = pd.get_dummies(data[['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'is_holiday']], drop_first=True)
# data = pd.concat([data, data_dummies], axis = 1)
# categorical_columns.remove('is_delay')
# data.drop(categorical_columns,axis=1, inplace=True)
# ```

# Check the new columns.

# In[303]:


data.columns


# ### Sample output
# 
# ```
# Index(['Distance', 'DepHourofDay', 'is_delay', 'AWND_O', 'PRCP_O', 'TAVG_O',
#        'AWND_D', 'PRCP_D', 'TAVG_D', 'SNOW_O', 'SNOW_D', 'Year_2015',
#        'Year_2016', 'Year_2017', 'Year_2018', 'Quarter_2', 'Quarter_3',
#        'Quarter_4', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
#        'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
#        'DayofMonth_2', 'DayofMonth_3', 'DayofMonth_4', 'DayofMonth_5',
#        'DayofMonth_6', 'DayofMonth_7', 'DayofMonth_8', 'DayofMonth_9',
#        'DayofMonth_10', 'DayofMonth_11', 'DayofMonth_12', 'DayofMonth_13',
#        'DayofMonth_14', 'DayofMonth_15', 'DayofMonth_16', 'DayofMonth_17',
#        'DayofMonth_18', 'DayofMonth_19', 'DayofMonth_20', 'DayofMonth_21',
#        'DayofMonth_22', 'DayofMonth_23', 'DayofMonth_24', 'DayofMonth_25',
#        'DayofMonth_26', 'DayofMonth_27', 'DayofMonth_28', 'DayofMonth_29',
#        'DayofMonth_30', 'DayofMonth_31', 'DayOfWeek_2', 'DayOfWeek_3',
#        'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
#        'Reporting_Airline_DL', 'Reporting_Airline_OO', 'Reporting_Airline_UA',
#        'Reporting_Airline_WN', 'Origin_CLT', 'Origin_DEN', 'Origin_DFW',
#        'Origin_IAH', 'Origin_LAX', 'Origin_ORD', 'Origin_PHX', 'Origin_SFO',
#        'Dest_CLT', 'Dest_DEN', 'Dest_DFW', 'Dest_IAH', 'Dest_LAX', 'Dest_ORD',
#        'Dest_PHX', 'Dest_SFO', 'is_holiday_1'],
#       dtype='object')
# ```

# Rename the `is_delay` column to `target` again. Use the same code as before.

# In[304]:


data.rename(columns = {'is_delay':'target'}, inplace=True )# Enter your code here


# In[305]:


data.head()


# In[306]:


# write code to Save the new combined csv file (combined_csv_v2.csv) to your local computer
# note this combined file will be also used in part B
data.to_csv('combined_csv_v2.csv', index=False)


# Create the training and testing sets again.

# In[307]:


# Enter your code here
from sklearn.model_selection import train_test_split

# Define the features (X) and target (y)
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# ### New baseline classifier
# 
# Now, see if these new features add any predictive power to the model.

# In[308]:


# Instantiate the logistic regression classifier
classifier2 = LogisticRegression()

# Fit the model to the training data
classifier2.fit(X_train, y_train)


# In[309]:


# Enter your code here# Make predictions on the test data
y_pred = classifier2.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)


# Perform the evaluaion as you have done with the previous model and plot/show the same metrics

# Question: did you notice a difference by adding the extra data on the results?

# Accuracy on the test set is approximately 0.7869, which is slightly lower than the previous accuracy. 
# Precision for class 0 (not delayed) is 0.79, while precision for class 1 (delayed) is 0.51. The precision for class 1 has decreased from the previous model, indicating that the new model still has a relatively high false positive rate for delay predictions.
# Recall for class 0 is 0.99, while recall for class 1 is only 0.03. The recall for class 1 has also decreased further, indicating that the new model continues to have a high rate of false negatives.
# F1-score for class 1 is only 0.06, which is extremely low, indicating that the model's ability to correctly classify delayed flights remains quite poor.
# 
# 
# Similar to the previous model, the recall for class 1 (delayed flights) is very low, which means that the model is still missing a large number of actual delayed flights. 
# 
# The confusion matrix still shows a large number of false negatives (56562), indicating that the model is missing a significant portion of actual delayed flights. This issue remains unchanged from the previous model.

# # Step 6: Using Tableau
# 
# Use Tableau to load the combined_csv_v2.csv file and build a dashboard that show your understanding of the data and business problem. 
# ### what to do:
# 1. Load the data into Tableau and build the dashboard
# 2. Share the dashboard on your Tableau public account 
# 3. Copy the link of the shared dashboard below
# 
# Note: The dashboard needs to be self explainable to others, so make it simple and add only the features that you feel heighlight the main question(s) of the prblem statement.

# 
# https://public.tableau.com/app/profile/himani.malik.khurana/viz/TableauDashboard_16989699214990/Dashboard1

# ## Conclusion
# 
# You've now gone through at least a couple iterations of training and evaluating your model. It's time to wrap up this project and reflect on what you've learned and what types of steps you might take moving forward (assuming you had more time). Use the cell below to answer some of these and other relevant questions:
# 
# 1. Does your model performance meet your business goal? If not, what are some things you'd like to do differently if you had more time for tuning?
# 2. To what extent did your model improve as you made changes to your dataset? What types of techniques did you employ throughout this project that you felt yielded the greatest improvements in your model?
# 3. What were some of the biggest challenges you encountered throughout this project?
# 4. What were the three most important things you learned about machine learning while completing this project?

# 1. The model's performance does not meet the business goal. There is very low recall for delayed flights which means model is missing a significant number of actual delayed flights.
# 2. There was no significant improvement in the model's performance as key metrics remained low.
# 3. One of the biggets challenges in this case in imbalanced dataset. This made it difficult for the model to learn from the minority class i.e delayed flights and resulted in low recall. 
# 4. Key learnings:
#     - High quality data is essential for building effective models. Inaccurate data can impact models' performance.
#     - There was imbalanced dataset which needs to be addressed before training the model to reduce bias in the model.
#     -  Domain knowledge matters as it helps in selecting relevant features and evaluating model results in a meaningful context.

# In[ ]:




