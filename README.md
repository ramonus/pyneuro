# Neural Network library

Python3 Neural Network module. It supports all networks size, each layer can have it's specific activation function, size, and configuration.
The module supports all sizes for networks.

The optimizer algorithm it's SGD and currently it's the only available.

## Usage
### 1. Clone this module:
```
 git clone https://github.com/ramonus/pyneuro.git
```
### 2. Install libraries:
```
pip3 install numpy
```
**or**
```
pip install numpy
```
### 3. Use it in your code:

   **3.1. To run the example just run the main file** `nn.py`
```
from pyneuro import *
import numpy as np

X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
Y = [
    [0],
    [1],
    [1],
    [0]
]
nn = input_data(2)
nn = fully_connected(nn,2,activation="sigmoid")
nn = fully_connected(nn,1,activation="sigmoid")
Yh = nn.predict(X)
nn.fit(X,Y,n_epochs=10000,learning_rate=0.5,verbose=100)
Yh2 = nn.predict(X)
print("Y:",Y)
print("Y before fit:",[np.round(i[0]).tolist() for i in np.divide(Yh,max(Yh))])
print("Y after fit :",[np.round(i[0]).tolist() for i in np.divide(Yh2,max(Yh2))])
print("             ",[i[0].tolist() for i in Yh2])
print("Done!")
```
