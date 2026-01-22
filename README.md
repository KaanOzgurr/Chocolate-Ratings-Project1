🍫 Chocolate Bar Ratings: Statistical Analysis & ML Modeling

This project provides a comprehensive data science pipeline to analyze factors defining premium chocolate. Using a dataset of over 2,500 chocolate bars, I explored the relationship between cocoa solids, geographical origin, and expert ratings.



🎯 Project Objectives

Data Cleaning: Standardized complex string data into model-ready numerical features.

Statistical Profiling: Identified if higher cocoa percentages correlate with better quality scores.

Predictive Modeling: Evaluated Linear and Ensemble models (Random Forest) to predict expert ratings.

Feature Selection: Used Recursive Feature Elimination (RFE) to quantify the impact of manufacturers versus bean origins.















🔍 Visual Insights & Analysis

1. Feature Correlations

The heatmap reveals a weak negative correlation (-0.15) between Cocoa Percent and Rating. This suggests that "darker" is not always better for expert scores. ![Correlation Matrix]  (./Correlation Matrix.png)

2. Rating Distribution

Expert ratings follow a Left-Skewed Distribution, peaking at the 3.0 - 3.5 range. Scores below 2.0 or at a perfect 5.0 are statistical outliers. ![Rating Distribution]   (./Rating Distribution.png)

3. Feature Importance

The Random Forest model identifies the Manufacturer (Company) as the most significant predictor of the final rating, followed by the Bean Origin. ![Feature Importance]   (./Feature Importance 2.png)

4. Top Performing Manufacturers

The analysis highlights top-tier companies, such as Pralus and Ocelot, which consistently achieve higher average ratings compared to others in the dataset. ![Top Companies]   (./Top Companies.png)






🛠️ Tech Stack

Language: Python 3.14

Libraries: Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib

Version Control: Git & GitHub












📈 Model Performance

Model,                    Mean Squared Error (MSE),              R² Score
Linear Regression,               0.2099,                          -0.0109
Random Forest,                   0.2110,                          -0.0165













## Dataset
This project uses a dataset originally published on Kaggle.
License: CC0 (Public Domain)

Although attribution is not required, the original source is acknowledged for transparency.
This dataset was sourced from Kaggle ( https://www.kaggle.com/datasets/andrewmvd/chocolate-ratings ) 
