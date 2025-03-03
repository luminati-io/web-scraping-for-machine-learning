# Web Scraping for Machine Learning

[![Promo](https://github.com/luminati-io/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.com/) 

This guide explains how to collect, prepare, and use web-scraped data for machine learning projects, including [ETL](https://brightdata.com/blog/proxy-101/etl-pipeline) setup and model training tips. Before you proceed further, we recommend you get more familiar with Python web scraping. 

- [Performing Scraping for Machine Learning](#performing-scraping-for-machine-learning)
- [Using Machine Learning on Scraped Data](#using-machine-learning-on-scraped-data)
- [Notes on Fitting an LSTM Neural Network](#notes-on-fitting-an-lstm-neural-network)
- [Setting Up ETLs When Scraping Data for Machine Learning](#setting-up-etls-when-scraping-data-for-machine-learning)

## What Is Machine Learning?

Machine learning (ML) is a branch of AI that enables systems to learn from data without explicit programming. It applies mathematical models to recognize patterns in data, allowing computers to make predictions based on new inputs.

## Why Web Scraping is Useful for Machine Learning

Machine learning and AI systems rely on data to train models, making web scraping a valuable tool for data professionals. Here is why web scraping is useful for ML:

- **Data collection at scale**: ML models, especially deep learning ones, require vast datasets. Web scraping enables large-scale data gathering.
- **Diverse and rich data sources**: The web provides a wide variety of data, enriching existing datasets for better model training.
- **Up-to-date information**: For models needing the latest trends (e.g., stock predictions, sentiment analysis), web scraping ensures fresh data.
- **Enhancing model performance**: More data improves model accuracy and validation, making web scraping a key resource.
- **Market analysis**: Extracting reviews, ratings, and trends aids in consumer sentiment analysis and business insights.

## Guide Prerequisites

To follow the guide, you need the following prerequisites in your system:

- Python 3.6 or newer
- Jupyter Notebook 6.x
- An IDE, such as VS Code

## Performing Scraping for Machine Learning

The step-by-step section explains how to scrape Yahoo Finance to get NVIDIA stock prices for maching learning.

### Step #1: Set up the environment

Create a repository that has the following subfolders: `data`, `notebooks`, and `scripts`.

```
scraping_project/
├── data/
│   └── ...
├── notebooks/
│   └── analysis.ipynb
├── scripts/
│   └── data_retrieval.py
└── venv/
```

In this project:

- `data_retrieval.py` will contain your scraping logic.
- `analysis.ipynb` will contain the maching learning logic.
- `data/` will contain the scraped data to analyze via maching learning.

Create the virtual environment:

```bash
python3 -m venv venv 
```

To activate it, on Windows, run:

```powershell
venv\Scripts\activate
```

On macOS/Linux, execute:

```bash
source venv/bin/activate
```

Install the libraries you will need:

```bash
pip install selenium requests pandas matplotlib scikit-learn tensorflow notebook
```

### Step #2: Define the target page

To get the NVIDIA historical data, you have to go to the following URL:

```
https://finance.yahoo.com/quote/NVDA/history/
```

The page presents has filters to define how you want the data to be displayed:

![filters that allow you to define how you want the data to be displayed](https://brightdata.com/wp-content/uploads/2024/11/image-53-1024x91.png)

To retrieve enough data for machine learning, you can filter them by 5 years. You can use this URL that includes the filter:

```
https://finance.yahoo.com/quote/NVDA/history/?frequency=1d&period1=1574082848&period2=1731931014
```

Now you have to target the following table and retrieve the data from it:

![Table with daily financial data like open and close price, low, high, and more](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-54.png)

The CSS selector that defines the table is `.table` so you can write the following code in the `data_retrieval.py` file:

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import NoSuchElementException
import pandas as pd
import os

# Configure Selenium
driver = webdriver.Chrome(service=Service())

# Target URL
url = "https://finance.yahoo.com/quote/NVDA/history/?frequency=1d&period1=1574082848&period2=1731931014"
driver.get(url)

# Wait for the table to load
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".table"))
    )
except NoSuchElementException:
    print("The table was not found, verify the HTML structure.")
    driver.quit()
    exit()

# Locate the table and extract its rows
table = driver.find_element(By.CSS_SELECTOR, ".table")
rows = table.find_elements(By.TAG_NAME, "tr")
```

The above code snippet does the following:

- Sets up a Selenium Chrome driver instance
- Defines the target URL and instruct Selenium to visit it
- Waits for the table to be loaded: In this case, the target table is loaded by Javascript, so the web driver waits 20 seconds, just to be sure the table is loaded
- Intercepts the whole table by using the dedicated CSS selector

### Step #3: Retrieve the data and save them into a CSV file

Now you need to extract the headers from the table, retrieve all the data from the table, and convert the data into a [Numpy data frame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html).

You can do this with the following code:

```python
# Extract headers from the first row of the table
headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

# Extract data from the subsequent rows
data = []
for row in rows[1:]:
    cols = [col.text for col in row.find_elements(By.TAG_NAME, "td")]
    if cols:
        data.append(cols)

# Convert data into a pandas DataFrame
df = pd.DataFrame(data, columns=headers)
```

### Step #4: Save the CSV file into the `data/` folder

The CVS file that the script generates has to be saved into the `data/` folder. Here is the code for that:

```python
# Determine the path to save the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))  

# Navigate to the "data/" directory
data_dir = os.path.join(current_dir, "../data") 

# Ensure the directory exists 
os.makedirs(data_dir, exist_ok=True)  

# Full path to the CSV file
csv_path = os.path.join(data_dir, "nvda_stock_data.csv")  

# Save the DataFrame to the CSV file
df.to_csv(csv_path, index=False)
print(f"Historical stock data saved to {csv_path}")

# Close the WebDriver
driver.quit()
```

This code determines the (absolute) current path using the method `os.path.dirname()`, navigates to the `data/` folder with the method `os.path.join()`, ensures it exists with the method `os.makedirs(data_dir, exist_ok=True)`, saves the data to a CSV file with the method `df.to_csv()` from the Pandas library, and finally quits the driver.

### Step #5: Putting it all together

Here is the complete code for the `data_retrieval.py` file:

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import NoSuchElementException
import pandas as pd
import os

# Configure Selenium
driver = webdriver.Chrome(service=Service())

# Target URL
url = "https://finance.yahoo.com/quote/NVDA/history/?frequency=1d&period1=1574082848&period2=1731931014"
driver.get(url)

# Wait for the table to load
try:
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.yf-j5d1ld.noDl"))
    )
except NoSuchElementException:
    print("The table was not found, verify the HTML structure.")
    driver.quit()
    exit()

# Locate the table and extract its rows
table = driver.find_element(By.CSS_SELECTOR, ".table")
rows = table.find_elements(By.TAG_NAME, "tr")

# Extract headers from the first row of the table
headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

# Extract data from the subsequent rows
data = []
for row in rows[1:]:
    cols = [col.text for col in row.find_elements(By.TAG_NAME, "td")]
    if cols:
        data.append(cols)

# Convert data into a pandas DataFrame
df = pd.DataFrame(data, columns=headers)

# Determine the path to save the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))  

# Navigate to the "data/" directory
data_dir = os.path.join(current_dir, "../data") 

# Ensure the directory exists 
os.makedirs(data_dir, exist_ok=True)

# Full path to the CSV file  
csv_path = os.path.join(data_dir, "nvda_stock_data.csv")

# Save the DataFrame to the CSV file
df.to_csv(csv_path, index=False)
print(f"Historical stock data saved to {csv_path}")

# Close the WebDriver
driver.quit()
```

On Windows, launch the above script with:

```powershell
python data_retrieval.py
```

On Linux/macOS:

```bash
python3 data_retrieval.py
```

Here is how the output scraped data appears:

![The output of the scraped table](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-55.png)

## Using Machine Learning on Scraped Data

Let's use the data in the CSV file in machine learning to make predictions.

### Step #1: Create a new Jupyter Notebook file

Navigate to the `notebooks/` folder from the main one:

```bash
cd notebooks 
```

Open a Jupyter Notebook:

```bash
jupyter notebook
```

When the browser is open, click on **New > Python3 (ipykernel)** to create a new Jupyter Notebook file:

![Creating a new Jupyter Notebook file](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-56.png)

Rename the file to `analysis.ipynb`.

### Step #2: Open the CSV file and show the head

Now you can open the CSV file containing the data and show the head of the data frame:

```python
import pandas as pd

# Path to the CSV file
csv_path = "../data/nvda_stock_data.csv"

# Open the CVS file
df = pd.read_csv(csv_path)

# Show head
df.head()
```

This code goes to the `data/` folder with `csv_path = "../data/nvda_stock_data.csv"`. Then, it opens the CSV with the method [`pd.read_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) as a data frame and shows its head (the first 5 rows) with the method [`df.head()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html).

This is the expected result:

![The expected result](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-57.png)

### Step #3: Visualize the trend over time of the `Adj Close` value

Now that the data frame is correctly loaded, you can visualize the trend of the `Adj Close` value, which represents the adjusted closing value:

```python
import matplotlib.pyplot as plt

# Ensure the "Date" column is in datetime forma
df["Date"] = pd.to_datetime(df["Date"])

# Sort the data by date (if not already sorted)
df = df.sort_values(by="Date")

# Plot the "Adj Close" values over time
plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Adj Close"], label="Adj Close", linewidth=2)

# Customize the plot
plt.title("NVDA Stock Adjusted Close Prices Over Time", fontsize=16) # Sets title
plt.xlabel("Date", fontsize=12) # Sets x-axis label
plt.ylabel("Adjusted Close Price (USD)", fontsize=12) # Sets y-axis label
plt.grid(True, linestyle="--", alpha=0.6) # Defines styles of the line
plt.legend(fontsize=12) # Shows legend
plt.tight_layout()

# Show the plot
plt.show()
```

The above code does the following:

- `df["Date"]` accesses the `Date` column of the data frame and, with the method `pd.to_datetime()`, ensures that the dates are in the date format
- The `df.sort_values()` sorts the dates of the `Date` column. This ensures the data will be displayed in chronological order.
- `plt.figure()` sets the dimensions of the plot and `plt.plot()` displays it
- The lines of code under the `# Customize the plot` comment are useful to customize the plot by providing the title, the labels of the axes, and displaying the legend
- The `plt.show()` method is the one that actually allows the plot to be displayed

The expected result is something like that:

![NVDA stock adjusted close prices over time example](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-58.png)

This plot shows the actual trend of the adjusted closed values over time of the NVIDIA stocks values. The machine learning model you will be training will have to predict them as best as it can.

### Step #3: Preparing data for machine learning

Let's clean up and prepare the data:

```python
from sklearn.preprocessing import MinMaxScaler

# Convert data types
df["Volume"] = pd.to_numeric(df["Volume"].str.replace(",", ""), errors="coerce")
df["Open"] = pd.to_numeric(df["Open"].str.replace(",", ""), errors="coerce")

# Handle missing values 
df = df.infer_objects().interpolate() 

# Select the target variable ("Adj Close") and scale the data
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data between 0 and 1
data = scaler.fit_transform(df[["Adj Close"]])
```

The above code does the following:

- Converts the `Volume` and `Open` values with the method `to_numeric()`
- Handles missing values by using interpolation to fill them with the method `interpolate()`
- Scales the data with the `MinMaxScaler()`
- Selects and transforms (scales it) the target variable `Adj Close` with the method `fit_transform()`

### Step #4: Create the train and test sets

The model used for this tutorial is an LSTM ([Long Short-Term Memory](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)), which is a RNN ([Recurrent Neural Network](https://www.ibm.com/topics/recurrent-neural-networks)). You need to create a sequence of steps to allow it to learn the data:

```python
import numpy as np

# Create sequences of 60 time steps for prediction
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(data)):
    X.append(data[i - sequence_length:i, 0])  # Last 60 days
    y.append(data[i, 0])  # Target value

X, y = np.array(X), np.array(y)

# Split into training and test sets
split_index = int(len(X) * 0.8)  # 80% training, 20% testing
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
```

The above code snippet:

- Creates a sequence of 60 time steps. `X` is the array of the features, `y` is the array of the target value.
- Splits the initial data frame: 80% becomes the train set, 20% becomes the test set.

### Step #5: Train the model

Let's train the RNN on the train set:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Reshape X for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the Sequential Neural Network
model = Sequential()
model.add(LSTM(32, activation="relu", return_sequences=False))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
```

This code does the following:

- Respahes the array of the features to be ready for the LSTM neural network by using the method `reshape()`, both for the train and test sets
- Builds the LSTM neural network by setting its parameters
- Fits the LSTM to the train set by using the method `fit()`

In other words, the model has now fitted the train set and it is ready to make predictions.

### Step #6: Make predictions and evaluate the model performance

Let's evaluate the model's performance:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make Predictions
y_pred = model.predict(X_test)

# Inverse scale predictions and actual values for comparison
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print results
print("\nLSTM Neural Network Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
```

This code does the following:

- Inverses the values on the horizontal axis so that the data can be lately presented in chronological order. This is done with the method `inverse_transform()`.
- Evaluates the model by using the [mean squared error](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.mean_squared_error.html) and the [R^2 score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.r2_score.html).

Statistical errors are possible due to the stochastical nature of ML models. Here is the expected result:

![Expected result considering the statistical errors](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-59.png)

These results indicate that the model is good to predict the `Adj Close`.

### Step #7: Compare actual vs predicted values with a plot

Comparing results using machine learning isn't always sufficient. Let's create a plot that compares the actual values of the `Adj Close` with the predicted ones by the LSTM model:

```python
# Visualize the Results
test_results = pd.DataFrame({
    "Date": df["Date"].iloc[len(df) - len(y_test):],  # Test set dates
    "Actual": y_test.flatten(),
    "Predicted": y_pred.flatten()
})

# Setting plot
plt.figure(figsize=(12, 6))
plt.plot(test_results["Date"], test_results["Actual"], label="Actual Adjusted Close", color="blue", linewidth=2)
plt.plot(test_results["Date"], test_results["Predicted"], label="Predicted Adjusted Close", color="orange", linestyle="--", linewidth=2)
plt.title("Actual vs Predicted Adjusted Close Prices (LSTM)", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Adjusted Close Price (USD)", fontsize=12)
plt.legend()
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()
```

This code:

- Sets the comparison of the actual and predicted values on the level of the test set, so the actual values have to be trimmed to the shape that the test set has. This is done with the methods `iloc()` and `flatten()`.
- Creates the plot, adds labels to the axes, and the title, and manages other settings to improve the visualization.

The expected result is something like this:

![Actual vs predicted adjusted close prices ](https://github.com/luminati-io/web-scraping-for-machine-learning/blob/main/images/image-60.png)

As the plot illustrates, the LSTM neural network's predicted values (yellow dotted line) closely match the actual values (solid blue line). While the analytical results were promising, the visualization further confirms their accuracy.

### Step #8: Putting it all together

Here is the complete code for the `analysis.ipynb` notebook:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Path to the CSV file
csv_path = "../data/nvda_stock_data.csv"  
# Open CSV as data frame
df = pd.read_csv(csv_path)

# Convert "Date" to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values(by="Date")

# Convert data types
df["Volume"] = pd.to_numeric(df["Volume"].str.replace(",", ""), errors="coerce")
df["Open"] = pd.to_numeric(df["Open"].str.replace(",", ""), errors="coerce")

# Handle missing values 
df = df.infer_objects().interpolate()

# Select the target variable ("Adj Close") and scale the data
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data between 0 and 1
data = scaler.fit_transform(df[["Adj Close"]])

# Prepare the Data for LSTM
# Create sequences of 60 time steps for prediction
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(data)):
    X.append(data[i - sequence_length:i, 0])  # Last 60 days
    y.append(data[i, 0])  # Target value

X, y = np.array(X), np.array(y)

# Split into training and test sets
split_index = int(len(X) * 0.8)  # 80% training, 20% testing
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape X for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the Sequential Neural Network
model = Sequential()
model.add(LSTM(32, activation="relu", return_sequences=False))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make Predictions
y_pred = model.predict(X_test)

# Inverse scale predictions and actual values for comparison
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nLSTM Neural Network Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Visualize the Results
test_results = pd.DataFrame({
    "Date": df["Date"].iloc[len(df) - len(y_test):],  # Test set dates
    "Actual": y_test.flatten(),
    "Predicted": y_pred.flatten()
})

# Setting plot
plt.figure(figsize=(12, 6))
plt.plot(test_results["Date"], test_results["Actual"], label="Actual Adjusted Close", color="blue", linewidth=2)
plt.plot(test_results["Date"], test_results["Predicted"], label="Predicted Adjusted Close", color="orange", linestyle="--", linewidth=2)
plt.title("Actual vs Predicted Adjusted Close Prices (LSTM)", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Adjusted Close Price (USD)", fontsize=12)
plt.legend()
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()
```

This code goes straight to the goal, skipping initial data previews and plotting only `Adj Close` values, as these steps were covered earlier for preliminary analysis.

> **Note**:
> 
> While the code is shown in parts, it's best to run the full code at once due to ML's stochastic nature; otherwise, the final plot may vary significantly.

## Notes on Fitting an LSTM Neural Network

For simplicity, this guide focuses directly on fitting an LSTM neural network. However, in real-world ML applications, the process involves several key steps:

1. **Preliminary Data Analysis**. This is the most crucial step, where you understand your data, clean NaN values, handle duplicates, and resolve any mathematical inconsistencies.

2. **Training ML Models**. The first model you try may not be the best. A common approach is [spot-checking](https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/), which involves:

   - Training 3-4 ML models on the training set and evaluating their performance.
   - Selecting the top 2-3 models and [tuning their hyperparameters](https://scikit-learn.org/1.5/modules/grid_search.html).
   - Comparing the best-tuned models on the test set.
   - Choosing the highest-performing model.

3. **Deployment**. The best-performing model is then deployed for production use.

## Setting Up ETLs When Scraping Data for Machine Learning

Saving web-scraped data as a CSV is a common practice in machine learning, especially at the start of a project when searching for the best predictive model.  

Once the best model is identified, an **ETL (Extract, Transform, Load) pipeline** is typically set up to automate data retrieval, cleaning, and storage.

Here is the ETL Process for ML Workflows:

- **Extract**: Retrieve data from various sources, including web scraping.  
- **Transform**: Clean and prepare the collected data.  
- **Load**: Store the processed data in a database or data warehouse.  

Once stored, the data is integrated into the ML workflow to **re-train and re-validate the model** with new data.

## Conclusion

Need data for machine learning but not familiar with web scraping?  [Check out our solutions for efficient data retrieval](https://brightdata.com/use-cases/data-for-ai).  

Sign up for a free Bright Data account to try our scraper APIs or explore our datasets.
