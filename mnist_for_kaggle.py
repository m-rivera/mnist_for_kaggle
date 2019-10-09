#!/usr/bin/env python
"""Image recognition project with MNIST"""
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")/255.0

# separate input and output
y_train = train["label"]
x_train = train.drop(labels=["label"],axis = 1)/255.0

# reshape
x_train = x_train.values.reshape(-1,28,28)
test_input = test.values.reshape(-1,28,28)

# set up model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

# compile
model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))

# fit
model.fit(x_train, y_train, epochs=10)

# predict
predictions = model.predict(test_input)

# format results
out_nums = np.array([np.argmax(i) for i in predictions])
image_ids = np.arange(1,len(out_nums)+1)
out_array = np.array([image_ids,out_nums]).T

np.savetxt("results.csv",out_array,delimiter=",",fmt="%1d",header="ImageId,Label", comments="")
