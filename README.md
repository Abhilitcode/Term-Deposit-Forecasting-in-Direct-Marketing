# Term Deposit Forecasting in Direct Marketing

## Project Overview

This project focuses on the prediction of whether a client will subscribe to a term deposit based on the data obtained from direct marketing campaigns (phone calls) of a Portuguese banking institution. The dataset used for this project contains 45,211 instances and 16 features, where the goal is to build a classification model to predict the likelihood of a client subscribing to a term deposit.

## Dataset Information

The data is related to direct marketing campaigns conducted by a Portuguese banking institution. The marketing campaigns were based on phone calls, and multiple contacts with the same client were often required to determine if they would subscribe to the bank's term deposit.

**Dataset Characteristics:**

- **Type:** Multivariate
- **Subject Area:** Business
- **Associated Tasks:** Classification
- **Feature Type:** Categorical, Integer
- **Number of Instances:** 45,211
- **Number of Features:** 16

**Available Datasets:**

1. **bank-additional-full.csv**: Contains all 41,188 examples with 20 inputs, ordered by date (from May 2008 to November 2010).
2. **bank-additional.csv**: Contains 10% of the examples (4,119), randomly selected from the first dataset, with 20 inputs.
3. **bank-full.csv**: Contains all examples and 17 inputs, ordered by date (older version with fewer inputs).
4. **bank.csv**: Contains 10% of the examples and 17 inputs, randomly selected from the third dataset.

The primary dataset used for this project is **bank-full.csv**.

## Project Objective

The objective of this project is to develop a robust machine-learning model that predicts whether a client will subscribe to a term deposit based on the provided features. This will help financial institutions in targeted marketing efforts, allowing them to focus on clients who are more likely to subscribe, thereby improving the efficiency of their campaigns.

## Project Workflow

1. **Data Preprocessing:** 
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling and normalization

2. **Exploratory Data Analysis (EDA):**
   - Understanding the distribution of variables
   - Visualizing relationships between features
   - Identifying patterns and insights

3. **Model Development:**
   - Splitting data into training and testing sets
   - Building various classification models (e.g., Logistic Regression, Random Forest, XGBoost)
   - Hyperparameter tuning using RandomizedSearchCV
   - Evaluating model performance using metrics like accuracy, precision, recall, and F1-score

4. **Model Deployment:**
   - Deploying the final model using Streamlit for a user-friendly interface
   - Dockerizing the application for containerization and easy deployment
   - Hosting the application on AWS EC2

## Tools and Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Model Deployment:** Streamlit, Docker, AWS EC2
- **Version Control:** GitHub

## Results

The final model achieved an accuracy of 89% using XGBoost. The model was deployed successfully on Streamlit with Docker and hosted on AWS EC2, providing an interactive interface for users to predict the likelihood of a client subscribing to a term deposit.

## Repository Structure

```plaintext
├── .github
├── Dockerfile
├── README.md
├── Term Deposit Forecasting in Direct Marketing.ipynb
├── app.py
├── bank-additional-full.csv
├── data_description.py
├── feature_dict.json
├── requirements.txt
└── xgboost_model

```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/abhilitcode/term-deposit-forecasting-in-direct-marketing.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app/streamlit_app.py
   ```
