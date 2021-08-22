import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('customer_churn.csv')

print(df.sample(5))
print(df.shape)

df.drop('customerID', axis='columns', inplace=True)

print(df.sample(5))

print(df.dtypes)

# in monthly charges the values are float but in total charges the values are objects

print(df.MonthlyCharges.values)
print(df.TotalCharges.values)

# coverting object to values
# pd.to_numeric(df.TotalCharges)  # if there is any spaces it will not convert throws error

# to see which rows are having spaces

# print(df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()])

# we need to drop that rows

df1 = df[df.TotalCharges != ' ']
print(df1.shape)
print(df.dtypes)
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
print(df1.dtypes)

# handling tenuer - how many years or month the customer is for that buisness

# we can find the person leaving or stay if churn yes means person leave else stay

tenure_churn_no = df[df.Churn == 'No'].tenure  # churn no for tenure
tenure_churn_yes = df[df.Churn == 'Yes'].tenure

plt.hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=['Churn Yes', 'Churn No'])
plt.legend()
plt.show()

# there many colums with categorical we need label encoding

def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':  # to remove the numbers
            print(f'{column}:{df[column].unique()}')  # printing only categorical columns

print_unique_col_values(df1)

# replacing no services to no
df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)
print_unique_col_values(df1)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

# print unique values in columns of yes no

for col in yes_no_columns:
    print(f'{col}:{df1[col].unique()}')

# replacing male and female to 0 and 1

df1['gender'].replace({"Male": 1, "Female": 0}, inplace=True)
print(df1['gender'].unique())

# one hot encoding to other column which has morethan two classes

df2 = pd.get_dummies(df1, columns=['InternetService', 'Contract', 'PaymentMethod'])

# scalling column values which is not range btw 0 and 1

col_scal = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()

df2[col_scal] = scaler.fit_transform(df2[col_scal])

for col in df2:
    print(f'{col}:{df2[col].unique()}')


# Build a model
X = df2.drop(['Churn'], axis='columns')
y = df2['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),  # shape 26-input layer, 20-hidden layer
    keras.layers.Dense(10, activation='relu'),   # 10-hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # output layers
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100)

print(model.evaluate(X_test, y_test))

yp = model.predict(X_test)


print(y_test[:10])

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(y_pred[:10])

# classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
