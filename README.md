# Traffic Flow Prediction Using Linear Regression and Decision Tree Classification

## 1. Project Overview

Traffic congestion is a common problem in urban areas. It can lead to longer travel times, increased fuel consumption, air pollution, and inefficient use of road infrastructure. Machine learning can help analyze historical traffic data and predict traffic conditions.

This project uses the **Metro Interstate Traffic Volume dataset** to develop a machine learning system for traffic flow prediction. It implements two algorithms: Linear Regression to predict numerical traffic volume and Decision Tree Classification to categorize traffic conditions into Low, Moderate, and High levels.

The project demonstrates how machine learning techniques can be applied to transportation data to understand traffic patterns and support traffic management.

## 2. Objectives

- Analyze historical traffic volume and related weather and time information.
- Perform data cleaning and preprocessing.
- Explore traffic patterns using data visualization.
- Predict numerical traffic volume using Linear Regression.
- Classify traffic conditions using a Decision Tree Classifier.
- Evaluate the models using appropriate performance metrics.
- Understand the application of machine learning in traffic analysis.

## 3. Dataset

**Dataset name:** Metro Interstate Traffic Volume

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

The dataset contains hourly traffic volume observations for westbound traffic on Interstate 94 in Minnesota, USA, along with weather conditions and date-time information.

### Important Features

| Feature | Description |
|---|---|
| `holiday` | Indicates whether the observation falls on a holiday |
| `temp` | Temperature |
| `rain_1h` | Rainfall recorded over one hour |
| `snow_1h` | Snowfall recorded over one hour |
| `clouds_all` | Cloud coverage |
| `weather_main` | Main weather condition |
| `weather_description` | Detailed weather description |
| `date_time` | Date and time of the observation |
| `traffic_volume` | Number of vehicles recorded during the hour |

The `traffic_volume` column is the target for the regression task. For classification, traffic volume is converted into three categories: Low, Moderate, and High.

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

The classification thresholds are derived from the training data to reduce data leakage from the test set.

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

## 6. Project Workflow

1. Load the Metro Interstate Traffic Volume dataset.
2. Inspect the dataset, including its columns, data types, and missing values.
3. Clean the data and handle invalid or missing records.
4. Perform exploratory data analysis using plots and visualizations.
5. Extract useful time-based features from the date-time column.
6. Separate input features and target variables.
7. Split the data into training and testing sets in chronological order.
8. Train the Linear Regression model for numerical traffic volume prediction.
9. Create traffic categories and train the Decision Tree Classifier.
10. Evaluate both models using suitable metrics.
11. Visualize the results and summarize the findings.

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

The actual results are reported after running the notebook.

## 8. How to Run the Project

This project is implemented in a Jupyter Notebook and can be executed using Google Colab.

1. Clone or download this GitHub repository.
2. Open `Traffic_Flow_Prediction.ipynb` in Google Colab.
3. Obtain the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume).
4. Upload the dataset CSV file when prompted by the notebook, or configure the dataset path as instructed in the notebook.
5. Run the notebook cells sequentially from beginning to end.
6. Review the visualizations, model evaluation metrics, and final results.

## 9. Expected Outcomes

The project is expected to demonstrate:

- Patterns in traffic volume across different times and weather conditions.
- Numerical traffic volume predictions from Linear Regression.
- Classification of traffic conditions into Low, Moderate, and High categories using a Decision Tree.
- Evaluation of the two models using appropriate metrics.
- The potential use of machine learning for traffic analysis and transportation planning.

Actual performance depends on the data, preprocessing, selected features, and model configuration.

## 10. Limitations

- The dataset represents a particular highway location and may not generalize to all roads or cities.
- Traffic conditions can be influenced by factors not included in the dataset, such as accidents, roadworks, and special events.
- Linear Regression may not capture all complex, nonlinear relationships in traffic data.
- Classification performance depends on the traffic-category thresholds and the distribution of the data.

## 11. Conclusion

This project explores the use of Linear Regression and Decision Tree Classification for traffic flow analysis. Linear Regression predicts numerical traffic volume, while Decision Tree Classification identifies traffic conditions as Low, Moderate, or High. By combining data preprocessing, visualization, model development, and evaluation, the project demonstrates a practical application of machine learning to a transportation problem.

## 12. Author

**Project:** Traffic Flow Prediction Using Linear Regression and Decision Tree Classification

Author:Varada K N
