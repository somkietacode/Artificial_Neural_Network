# Artificial Neural Network
![Artificial Neural Network logo](image/gitban.png)

This repository hosts the development of the Artificial Neural Network library.

## About Artificial Neural Network

Artificial Neural Network, is a deep learning API written in Python.
It was developed with a focus on enabling fast experimentation.
*Being able to go from idea to result as fast as possible is key to doing good research.*

Artificial Neural Network is:

-   **Simple** 
-   **Flexible** 
-   **Powerful** 

## First contact with Artificial Neural Network

The core data structures of Artificial Neural Network are __consign__ and __result__.
It implement four model in two layer neural network for helping you fast build __predictor__.

Here is an `exemple` :

```python
from Artificial_Neural_Network import artificialneuralnetwork_classifier
import pandas as pd
import numpy as np

# Reading and cleaning dataset form a CSV file

df = pd.read_csv('admission_data.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Select X dataset (consign) and convert them in numpy matrix 
x = np.matrix(df[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA"]].to_numpy() )

# Select Y dataset (response) and convert them in numpy matrix 
y = np.matrix(df[["Research"]].to_numpy())

# Train the model
ANN = artificialneuralnetwork_classifier(x,y)

```

Let make prediction

```python

X = np.matrix([[318,110,3,4,3,8.8] ])
print(Ann.predict(X))

```

It is a binairy classifier. Mean that your response should be 0 or 1. And your dataset response may also be binary.


