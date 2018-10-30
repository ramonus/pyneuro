**Neural Network library**

Python3 Neural Network module. It supports all networks size, each layer can have it's specific activation function, size, and configuration.
The module supports all sizes for networks.

The optimizer algorithm it's SGD and currently it's the only available.

**Usage**
1. Clone this module:
```
 git clone https://github.com/ramonus/pyneuro.git
```
2. Install libraries:
```
pip3 install numpy
```
or
```
pip install numpy
```
3. Use it with you'r programs:
```
from pyneuro import *
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
    nn = fully_connected(nn,2)
    nn = fully_connected(nn,1)
    Yh = nn.predict(X)
    print("Y before fit:",Yh)
    nn.fit(X,Y,1)
    Yh2 = nn.predict(X)
    print("Y after fit :",Yh2)
```
