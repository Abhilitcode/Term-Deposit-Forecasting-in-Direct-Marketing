import streamlit as st
import pandas as pd
import joblib
import json

best_model = joblib.load('xgboostt_model.joblib')

# Load the feature dictionary
with open('feature_dict.json', 'r') as fp:
    feature_dict = json.load(fp)

st.title("Term Deposit Forecasting in Direct Marketing")

#1
age = st.number_input('Age',min_value=0,max_value=120,step=1,format='%d')

#2
job_choices = ['Select a Job','admin.','blue-collar','entrepreneur','housemaid','management','retired',
                           'self-employed','services','student','technician','unemployed','unknown']

job = st.selectbox('Job',job_choices)

if job != "Select a Job":
    st.write(f"You selected {job} as your job.")
else:
    st.write("Please select the job from dropdown.")

#3
marital = st.radio("Marital Status",['single','married','divorced','unknown'])

#4
# Define a mapping of user-friendly names to original data values
education_mapping = {
    "Select your education level":"placeholder",
    "Basic (4 years)": "basic.4y",
    "Basic (6 years)": "basic.6y",
    "Basic (9 years)": "basic.9y",
    "High School": "high.school",
    "Illiterate": "illiterate",
    "Professional Course": "professional.course",
    "University Degree": "university.degree",
    "Unknown": "unknown"
}

education_level = st.selectbox("Education Level",list(education_mapping.keys()))

if education_level != "Select your education level":
    st.write(f"You selected {education_level} as your education.")
else:
    st.write("Please select your education level from dropdown.")

#5
default = st.radio("Has Credit Default ?",['yes','no','unknown'])
#6
housing = st.radio("Has Housing Loan ?",['yes','no','unknown'])
#7
personal = st.radio("Has Personal Loan ?",['yes','no','unknown'])

#8
contact_choices = ['Select your contact','cellular','telephone']

contact = st.selectbox('Communication Type',contact_choices)

if contact != "Select your contact":
    st.write(f"You selected {contact} as your communication type.")
else:
    st.write("Please select the contact from dropdown.")

#9
month_choices = ['Select last contact month of year','jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

month = st.selectbox('Last Contacted Month',month_choices)

if month != "Select last contact month of year":
    st.write(f"You selected {month} as your last contacted month.")
else:
    st.write("Please select the month from dropdown.")

#10
day_choices = ['Select last contact day of the week','mon', 'tue', 'wed', 'thu', 'fri']

day = st.selectbox('Last Contacted Day',day_choices)

if day != "Select last contact day of the week":
    st.write(f"You selected {day} as your last contacted day.")
else:
    st.write("Please select the day from dropdown.")
#11
duration = st.number_input("Duration of Last Contact (seconds)", min_value=0, step=1)
#12
campaign = st.number_input("Number of Contacts Performed During This Campaign", min_value=0, step=1)
#13
pdays = st.number_input("Number of Days Since Last Contact (Previous Campaign)", min_value=-1,step=1)
#14
previous = st.number_input("Number of Contacts Performed Before This Campaign", min_value=0,step=1)

#15
outcome_choices = ['Select your previous outcome','success','failure', 'nonexistent']

poutcome = st.selectbox("Outcome of Previous Campaign",outcome_choices)

if poutcome != "Select your previous outcome":
    st.write(f"You selected {poutcome} as your previous outcome.")
else:
    st.write("Please select the previous outcome from dropdown.")

#16

emp_var_rate = st.number_input("Employment Variation Rate")

#17
cons_price_idx = st.number_input("Consumer Price Index")

#18
cons_conf_idx = st.number_input("Consumer Confidence Index")

#19
euribor3m = st.number_input("3-Month Euribor Rate")

#20
nr_employed = st.number_input("Number of Employees")

#21
if st.button("Make Predictions"):
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education_level],
        'default': [default],
        'housing': [housing],
        'loan': [personal],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed],
    })
    
    numerical_data = input_data.select_dtypes(include='number')
    numerical_features = numerical_data.columns.to_list()

    categorical_data = input_data.select_dtypes(exclude='number')
    categorical_features = categorical_data.columns.to_list()
    

    X_test = {}
    
    # add numerical columns in X_test
    for col in numerical_features:
        X_test[col] = input_data[col].iloc[0]
    
    # add categorical columns in X_test
    for i in feature_dict:
        for j in feature_dict[i]:
            if input_data[i].iloc[0] == j:
                X_test[f'{i}_{j}'] = [1]
            else:
                X_test[f'{i}_{j}'] = [0]

    X_test_df = pd.DataFrame.from_dict(X_test)
    print(X_test_df.columns)
    print(X_test)
    X_test_df = X_test_df[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_basic.4y', 'education_basic.6y',
       'education_basic.9y', 'education_high.school', 'education_illiterate',
       'education_professional.course', 'education_university.degree',
       'education_unknown', 'default_no', 'default_unknown', 'default_yes',
       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
       'loan_unknown', 'loan_yes', 'contact_cellular', 'contact_telephone',
       'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',
       'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure',
       'poutcome_nonexistent', 'poutcome_success']]
    
# dont do drop_first in training
    predictions = best_model.predict(X_test_df)
    # st.write("Term Deposit")
    if predictions[0] == 0:
        st.write('No Term Deposit')
    else:
        st.write("Term Deposit")



