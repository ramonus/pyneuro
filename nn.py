import numpy as np
import time

def sigmoid(x,deriv=False):
    if deriv:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))
        
def add_before(ls,i):
    if type(ls)!=list:
        ls = ls.ravel().tolist()
    return [i]+ls
def add_after(ls,i):
    if type(ls)==np.ndarray:
        ls = ls.ravel().tolist()
    return ls+[i]

def matrixy(arr):
    x = arr
    if type(x)==np.ndarray:
        x = x.ravel().tolist()
    return np.array(x).reshape(1,len(x))
class NeuralNetwork:
    def __init__(self,layers,activation=sigmoid):
        if len(layers)<3:
            raise ValueError("Layers must be minimum 3")
        self.activation = activation
        self.layers = layers
        self.W = None
        self.Z = None
        self.A = None
        self._init_weight_matrix()
    @staticmethod
    def load_from_weights(w):
        W = [np.array(e) for e in w]
        layers = []
        for i in range(len(w)-1):
            layers.append(len(W[i]))
        layers.append(len(w[-1]))
        nn = NeuralNetwork(layers)
        nn.W = W
        return nn          
    def _init_weight_matrix(self):
        self.W = []
        i_layer = self.layers[0]
        o_layer = self.layers[-1]
        h_layers = self.layers[1:-1]
        i = 0
        fr = i_layer+1
        np.random.seed(int(time.time()))
        for i in range(len(self.layers)-1):
            self.W.append(2*np.random.rand(self.layers[i],self.layers[i+1])-1)

    def forward(self,X):
        self.Z = []
        self.A = []
        A = matrixy(X)
        self.A.append(A)
        for i,w in enumerate(self.W):
            z =A@w
            self.Z.append(matrixy(z))
            A = self.activation(z)
            self.A.append(matrixy(A))
        return A

class Trainer:
    def __init__(self,nn):
        self.nn = nn
        self.trainfit = []
    def train(self,xdata,ydata,iters=1,learning_rate=0.1):
        for i in range(iters):
            # if i%20==0:
            print("Epoch:",i)
            self._train(xdata,ydata,iters,learning_rate)
    def _train(self,xdata,ydata,iters=1,learning_rate=0.1):
        if len(xdata)!=len(ydata):
            raise ValueError("Data input and output are different size!")
        
        self.deltas = []
        self.dW = []
        trains = 0
        for i,(X,Y) in enumerate(zip(xdata,ydata)):
            X,Y = np.array(X).reshape(1,self.nn.layers[0]),np.array(Y).reshape(1,self.nn.layers[-1])
            yh = self.nn.forward(X) # get yh
            d = -(Y-yh)*self.nn.activation(self.nn.Z[-1],deriv=True)
            self.deltas = add_before(self.deltas,d)
            for j,w in enumerate(self.nn.W[-1::-1]):
                j = len(self.nn.W)-1-j
                A = self.nn.A[j]
                dw = A.T@d
                self.dW = add_before(self.dW,dw)
                if j != 0:
                    d = d@w.T*self.nn.activation(self.nn.Z[j-1],deriv=True)
                    self.deltas = add_before(self.deltas,d)
            self.update_weights(learning_rate)
            trains += (Y-yh)**2
        t = sum(trains)*0.5
        self.trainfit.append(t)
    def update_weights(self,learning_rate=0.1):
        for i,w in enumerate(self.nn.W):
            self.nn.W[i] = np.subtract(self.nn.W[i],self.dW[i]*learning_rate)

def main():
    # np.random.seed(0)
    layers = [2,5,1]
    nn = NeuralNetwork(layers)
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
    print("Before train")
    for i in X:
        print(nn.forward(i))

    t = Trainer(nn)
    t.train(X,Y,10000,learning_rate=0.1)
    
    print("After train")
    for i in X:
        print(nn.forward(i))
    print("Last error:",t.trainfit[-1])
    import matplotlib.pyplot as plt
    f,ax = plt.subplots(1,1)
    ax.grid()
    ax.plot(t.trainfit)
    plt.show()


if __name__=="__main__":
    main()