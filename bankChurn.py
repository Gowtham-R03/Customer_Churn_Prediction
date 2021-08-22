import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Churn_Modelling.csv')

print(df.sample(5))

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis='columns')
print(df.sample(5))

print(df.dtypes)

print(df['Geography'].unique())

dummies = pd.get_dummies(df.Geography)

df = pd.concat([df, dummies.drop(['Germany'], axis='columns')], axis='columns')

df['Gender'].replace({"Male": 1, "Female": 0}, inplace=True)

df = df.drop(['Geography'], axis='columns')

print(df.sample(5))
print(df.dtypes)
print(df.isnull().sum())

# scalling our values in range btw 0 and 1

col_scal = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

scaler = MinMaxScaler()

df[col_scal] = scaler.fit_transform(df[col_scal])

for col in df:
    print(f'{col}:{df[col].unique()}')


# Build a model

X = df.drop(['Exited'], axis='columns')
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print(X_train.shape)
print(y_test.shape)

model = keras.Sequential([
    keras.layers.Dense(6, input_shape=(11,), activation='relu'),  # shape 26-input layer, 20-hidden layer
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


print(y_test[:3])

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(y_pred[:3])

# classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


