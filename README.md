# Traffic Flow Prediction Using Linear Regression and Decision Tree Classification

## 1. Project Overview

Traffic congestion is a common problem in urban areas. It can lead to longer travel times, increased fuel consumption, air pollution, and inefficient use of road infrastructure. Machine learning can help analyze historical traffic data and predict traffic conditions.

This project uses the **Metro Interstate Traffic Volume dataset** to develop a machine learning system for traffic flow prediction. It implements two algorithms: Linear Regression to predict numerical traffic volume and Decision Tree Classification to categorize traffic conditions into Low, Moderate, and High levels.

The project has two parts:

- **A Jupyter / Google Colab notebook** that covers data cleaning, exploratory analysis, model training, and evaluation.
- **A Streamlit web app** that lets a user enter a date, time, and weather to get a traffic prediction, or forecast traffic for the coming days using a weather forecast.

The project demonstrates how machine learning techniques can be applied to transportation data to understand traffic patterns and support traffic management.

<!-- Live demo: add your Streamlit app link here after deploying -->

## 2. Objectives

- Analyze historical traffic volume and related weather and time information.
- Perform data cleaning and preprocessing.
- Explore traffic patterns using data visualization.
- Predict numerical traffic volume using Linear Regression.
- Classify traffic conditions using a Decision Tree Classifier.
- Evaluate the models using appropriate performance metrics.
- Predict traffic for new inputs and for upcoming days using weather forecast data.
- Provide an interactive web interface for making predictions.
- Understand the application of machine learning in traffic analysis.

## 3. Dataset

**Dataset name:** Metro Interstate Traffic Volume

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

The dataset contains hourly traffic volume observations for westbound traffic on Interstate 94 in Minnesota, USA (near Minneapolis-St. Paul), collected between 2012 and 2018, along with weather conditions and date-time information.

### Important Features

| Feature | Description |
|---|---|
| `holiday` | Indicates whether the observation falls on a holiday |
| `temp` | Temperature (Kelvin) |
| `rain_1h` | Rainfall recorded over one hour (mm) |
| `snow_1h` | Snowfall recorded over one hour (mm) |
| `clouds_all` | Cloud coverage (%) |
| `weather_main` | Main weather condition |
| `weather_description` | Detailed weather description |
| `date_time` | Date and time of the observation |
| `traffic_volume` | Number of vehicles recorded during the hour |

The `traffic_volume` column is the target for the regression task. For classification, traffic volume is converted into three categories: Low, Moderate, and High.

### Data Quality Issues Handled

- Duplicate timestamps are removed.
- Invalid temperature readings of 0 K and an impossible rainfall value are treated as missing and filled using training-set medians.
- The holiday value `"None"` is replaced with `"No Holiday"`.

## 4. Algorithms Used

### 4.1 Linear Regression

Linear Regression is used to predict traffic volume as a continuous numerical value based on selected traffic, weather, and time-related features.

**Input:** Selected features from the dataset.

**Output:** Predicted numerical traffic volume.

### 4.2 Decision Tree Classification

A Decision Tree Classifier is used to classify traffic conditions into three categories:

- **Low:** Relatively low traffic volume.
- **Moderate:** Intermediate traffic volume.
- **High:** Relatively high traffic volume.

The classification thresholds are the 33rd and 67th percentiles of the training data, so the test set does not influence them.

**Input:** Selected traffic, weather, and time-related features.

**Output:** Predicted traffic category.

## 5. Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit (web app)
- Altair (interactive charts in the web app)
- Requests (weather forecast data)
- Open-Meteo API (free weather forecast, no API key required)

## 6. Project Workflow

1. Load the Metro Interstate Traffic Volume dataset.
2. Inspect the dataset, including its columns, data types, and missing values.
3. Clean the data and handle invalid or missing records.
4. Perform exploratory data analysis using plots and visualizations.
5. Extract useful time-based features from the date-time column (hour, month, day of week, weekend and holiday indicators, and cyclical sine/cosine encodings).
6. Separate input features and target variables.
7. Split the data into training and testing sets in chronological order (earlier 80% for training, later 20% for testing).
8. Fit the preprocessing steps (median imputation, most-frequent imputation, one-hot encoding) on the training data only.
9. Train the Linear Regression model for numerical traffic volume prediction.
10. Create traffic categories and train the Decision Tree Classifier.
11. Evaluate both models using suitable metrics.
12. Visualize the results and summarize the findings.
13. Predict traffic for a user-entered date, time, and weather.
14. Forecast traffic for upcoming days using a weather forecast.
15. Serve the predictions through a Streamlit web app.

## 7. Evaluation Metrics

### Linear Regression

The regression model is evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted traffic volume.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Indicates how well the model explains variation in the target variable.

### Decision Tree Classification

The classification model is evaluated using:

- **Accuracy:** Proportion of correctly classified observations.
- **Precision:** Measures how many predictions for a class are correct.
- **Recall:** Measures how many actual observations of a class are identified correctly.
- **F1-score:** Combines precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each traffic category.

The actual results are reported after running the notebook. The web app also shows them on its **Model performance** tab.

## 8. Predicting Traffic for New Dates

The trained models need weather and time as input. The dataset (2012-2018) teaches the models how traffic relates to time and weather, but it cannot contain the weather of a future date. For future dates the project therefore combines two sources:

| Part | Role |
|---|---|
| Historical dataset | Trains the models so they learn traffic patterns |
| Weather forecast (Open-Meteo API) | Supplies the future temperature, rain, snow, cloud cover, and weather type |
| Trained models | Turn those inputs into a predicted traffic volume and level |

The forecast is requested for the dataset's road location (I-94 near Minneapolis-St. Paul), because that is the road the models learned from. Live forecasts are available up to 15 days ahead. For longer horizons the app can instead use typical weather for the same month and hour, calculated from the training data.

## 9. Project Structure

```
.
├── Traffic_Flow_Prediction.ipynb   # Notebook: cleaning, EDA, training, evaluation, predictions
├── app.py                          # Streamlit web app (user interface)
├── traffic_core.py                 # Cleaning, training, and prediction logic used by the app
├── requirements.txt                # Python packages for the web app
└── README.md
```

## 10. How to Run the Project

### 10.1 Notebook (Google Colab)

1. Clone or download this GitHub repository.
2. Open `Traffic_Flow_Prediction.ipynb` in Google Colab.
3. Run the notebook cells sequentially from beginning to end. The notebook downloads the dataset from the UCI Machine Learning Repository automatically. If the download fails, download the dataset from the [UCI page](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and upload the CSV file when the notebook asks for it.
4. Review the visualizations, model evaluation metrics, and final results.

### 10.2 Web App

The app downloads the dataset and trains both models automatically on the first run (this takes a few seconds). If the download fails, it asks you to upload the CSV.

**Option A: On your own computer**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

**Option B: In Google Colab**

Upload `app.py`, `traffic_core.py`, and `requirements.txt` to the Colab file panel, then run:

```python
!pip install -q streamlit
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```

```python
!streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false &>/content/streamlit_logs.txt &
!./cloudflared tunnel --url http://localhost:8501
```

Open the `https://....trycloudflare.com` link shown in the output. The link works only while the cell is running.

**Option C: Deploy for free on Streamlit Community Cloud**

1. Upload `app.py`, `traffic_core.py`, and `requirements.txt` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, choose **New app**, select the repository, and set the main file to `app.py`.
3. Click **Deploy** to get a public link.

### 10.3 Web App Features

| Tab | What it does |
|---|---|
| **Predict a moment** | Enter a date, hour, and weather to get the predicted traffic volume and level, plus an hour-by-hour chart for that day |
| **Forecast ahead** | Predicts traffic for the next 1 to 15 days using a live weather forecast, or up to 30 days using typical weather; shows a daily summary, hourly charts, and a CSV download |
| **Model performance** | Shows MAE, RMSE, R², accuracy, precision, recall, F1-score, the confusion matrix, and feature importance |
| **About** | Describes the method and limitations |

## 11. Expected Outcomes

The project is expected to demonstrate:

- Patterns in traffic volume across different times and weather conditions.
- Numerical traffic volume predictions from Linear Regression.
- Classification of traffic conditions into Low, Moderate, and High categories using a Decision Tree.
- Evaluation of the two models using appropriate metrics.
- Traffic predictions for new inputs and upcoming days through a web interface.
- The potential use of machine learning for traffic analysis and transportation planning.

Actual performance depends on the data, preprocessing, selected features, and model configuration.

## 12. Limitations

- The dataset represents a particular highway location and may not generalize to all roads or cities.
- The data covers 2012 to 2018, so predictions reflect traffic patterns from that period and may drift if traffic has changed since.
- Traffic conditions can be influenced by factors not included in the dataset, such as accidents, roadworks, and special events.
- Linear Regression may not capture all complex, nonlinear relationships in traffic data, such as the two daily rush-hour peaks.
- Classification performance depends on the traffic-category thresholds and the distribution of the data. Because the split is chronological, the class balance in the test set can differ from the training set.
- Predictions for future dates depend on weather forecasts, which become less accurate the further ahead they look, especially for rain and snow.
- The multi-day forecast does not take public holidays into account.
- No lag features (such as the previous hour's traffic) are used.

## 13. Conclusion

This project explores the use of Linear Regression and Decision Tree Classification for traffic flow analysis. Linear Regression predicts numerical traffic volume, while Decision Tree Classification identifies traffic conditions as Low, Moderate, or High. By combining data preprocessing, visualization, model development, evaluation, and a web interface that uses weather forecasts for future dates, the project demonstrates a practical application of machine learning to a transportation problem.

## 14. Author

**Project:** Traffic Flow Prediction Using Linear Regression and Decision Tree Classification

Author: Varada K N