# Codsoft
data science projects 

project 1 : Titanic survival prediction

🚢 Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using Python, Data Science, and Machine Learning techniques. It is a classic beginner-friendly dataset from Kaggle, widely used to practice classification problems.

📌 Project Overview

The goal of this project is to build a predictive model that answers the question:
“What sorts of people were more likely to survive the Titanic sinking?”

We use machine learning algorithms to analyze passenger details like age, gender, class, and more, to predict survival chances.

📂 Dataset

Dataset: Titanic Dataset - Kaggle

Key Features:

PassengerId – Unique ID of passenger

Pclass – Ticket class (1st, 2nd, 3rd)

Name, Sex, Age – Personal details

SibSp, Parch – Family onboard

Fare – Ticket fare

Embarked – Port of Embarkation

Target Variable:

Survived (0 = No, 1 = Yes)

🛠️ Tools & Libraries Used

Python (3.x)

Libraries:

pandas → data manipulation

numpy → numerical computations

matplotlib, seaborn → visualization

scikit-learn → machine learning models

joblib → saving & loading trained model

⚙️ Project Workflow

Data Preprocessing

Handle missing values

Encode categorical features (Sex, Embarked)

Feature scaling

Exploratory Data Analysis (EDA)

Visualize survival rates by gender, class, age, etc.

Detect correlations between features

Model Building

Tried multiple ML algorithms: Logistic Regression, Random Forest, Decision Tree, etc.

Evaluated using accuracy and classification metrics

Model Deployment 

Saved the trained model as .pkl file

Built a simple prediction script for new inputs

project 2 : movie rating prediction


🎬 Movie Rating Prediction

This project predicts IMDB-style movie ratings using Python, Data Science, and Machine Learning techniques. The dataset contains details about movies such as name, year, duration, genre, votes, directors, and actors. The goal is to train a machine learning model that can predict a movie’s rating based on its features.

📌 Project Overview

Movie ratings are influenced by multiple factors such as genre, duration, and cast. Using this dataset, we apply data preprocessing, feature engineering, and machine learning models to predict the numerical rating of movies.

This is a regression problem where the target variable is the Rating.

📂 Dataset

Example Dataset Columns:

Name – Movie title

Year – Release year

Duration – Movie length in minutes

Genre – Type of movie (Drama, Comedy, etc.)

Rating – IMDb-style rating (Target variable)

Votes – Number of votes received

Director – Movie director

Actor 1, Actor 2, Actor 3 – Main cast

🛠️ Tools & Libraries Used

Python (3.x)

Libraries:

pandas → data manipulation

numpy → numerical computations

matplotlib, seaborn → visualization

scikit-learn → preprocessing & ML models

joblib → saving trained model

⚙️ Project Workflow

Data Preprocessing

Handle missing values

Encode categorical features (Genre, Director, Actors) using Label Encoding

Convert duration into numeric format

Feature scaling for numerical data

Exploratory Data Analysis (EDA)

Rating distribution analysis

Correlation between votes, genre, and rating

Impact of cast/director on ratings

Model Building

Algorithms used: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor

Evaluation metrics: MAE, RMSE, R² Score



project 3 : Sales prediction
💰 Sales Prediction

This project predicts future sales for a retail or business dataset using Python, Data Science, and Machine Learning techniques. The goal is to analyze historical sales data and build a predictive model that helps businesses forecast sales accurately.

📌 Project Overview

Sales prediction is a crucial task for businesses to optimize inventory, manage resources, and plan marketing strategies. Using historical sales data, this project applies data preprocessing, feature engineering, and machine learning algorithms to predict sales for future periods.

This is a regression problem, where the target variable is the sales amount.

🛠️ Tools & Libraries Used

Python (3.x)

Libraries:

pandas → data manipulation

numpy → numerical computations

matplotlib, seaborn → data visualization

scikit-learn → preprocessing & ML models


task 4 : credit card fraud detection
This project aims to detect fraudulent credit card transactions using machine learning techniques. It leverages Python-based data science tools to build, evaluate, and deploy a predictive model that flags suspicious activity.

## 📊 Dataset 


The dataset used is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains transactions made by European cardholders in September 2013. It includes:
- 284,807 transactions
- 492 fraud cases (highly imbalanced)
- Features are anonymized (V1–V28) via PCA, plus `Time`, `Amount`, and `Class` (target)

## 🧠 Project Workflow

1. **Data Preprocessing**
   - Handle missing values (if any)
   - Normalize `Amount` and `Time`
   - Address class imbalance using techniques like SMOTE or undersampling

2. **Exploratory Data Analysis (EDA)**
   - Visualize distributions, correlations, and fraud patterns
   - Use `Seaborn`, `Matplotlib`, and `Pandas` profiling

3. **Model Building**
   - Algorithms: Logistic Regression, Random Forest, XGBoost, etc.
   - Evaluation metrics: Precision
   - 
4. **Model Evaluation**
   - Confusion matrix heatmap
   - Cross-validation
   - Feature importance analysis

