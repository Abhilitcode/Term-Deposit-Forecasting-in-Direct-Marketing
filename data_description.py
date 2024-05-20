import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)


df = pd.read_csv('bank-additional-full.csv', delimiter=';')
# df
# df.columns

df['y'] = df['y'].replace({'yes':1,'no':0})

X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
y = df['y']
X.select_dtypes('object')
X_dummies = pd.get_dummies(X)

numerical_data = X.select_dtypes(include='number')
numerical_features = numerical_data.columns.to_list()

categorical_data = X.select_dtypes(exclude='number')
categorical_features = categorical_data.columns.to_list()

feature_dict = {}
for i in categorical_features:
    unique_vals = list(X[i].unique())
    feature_dict[i] = unique_vals


with open('feature_dict.json', 'w') as fp:
    json.dump(feature_dict, fp)

# X_numerical = X_dummies.drop(columns=categorical_features)




# final_test_X = scaler.transform(X_test)
# output = model.predict(final_test_X)
