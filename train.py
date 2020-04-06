import pandas as pd
df = pd.read_csv('/content/breslow.csv')
print(df.head())

df = df.drop('Unnamed: 0', 1)
properties = list(df.columns.values)
properties.remove('ns')
print(properties)
X = df[properties]
y = df['ns']

print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

import tensorflow as tf
from tensorflow import keras
import torch
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

import pickle
import torch
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
#filename = 'final.sav'
#pickle.dump(model, open(filename, 'wb'))
model_save_name = 'final.pt'
path = F"/content/{model_save_name}" 
torch.save(model.state_dict(), path)
