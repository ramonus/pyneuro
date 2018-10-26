import numpy as np
import time, json
from tqdm import tqdm

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
    def __init__(self,layers,activation=sigmoid,biased=True):
        if len(layers)<3:
            raise ValueError("Layers must be minimum 3")
        self.activation = activation
        self.layers = layers
        self.W = None
        self.Z = None
        self.A = None
        self.biased = biased
        self._init_weight_matrix()
    @staticmethod
    def load_from_weights(data):
        w = data["w"]
        biased = data["biased"]
        W = [np.array(e) for e in w]
        layers = []
        for i in range(len(w)):
            layers.append(len(W[i]))
        layers.append(len(w[-1][0]))
        if biased:
            for i in range(len(layers)-2):
                layers[i] -= 1
        nn = NeuralNetwork(layers,biased=biased)
        nn.W = W
        return nn          
    def _init_weight_matrix(self):
        self.W = []
        np.random.seed(int(time.time()))
        for i in range(len(self.layers)-1):
            dim = [self.layers[i],self.layers[i+1]]
            if self.biased and i<len(self.layers)-2:
                dim[0] += 1
            self.W.append(2*np.random.rand(*dim)-1)

    def forward(self,X):
        self.Z = []
        self.A = []
        A = matrixy(X)
        self.A.append(A)
        if self.biased:
            A = matrixy(add_after(A,1))
        for i,w in enumerate(self.W):
            z =A@w
            self.Z.append(matrixy(z))
            A = self.activation(z)
            self.A.append(matrixy(A))
            if i<len(self.W)-2 and self.biased:
                A = matrixy(add_after(A,1))            
        return A
    def save_to_file(self,fn):
        try:
            if not fn.endswith(".json"):
                fn += ".json"
            data = {
                "w":[i.tolist() for i in self.W],
                "biased": self.biased
            }
            with open(fn,"w") as f:
                f.write(json.dumps(data))
        except Exception as e:
            raise e
    @staticmethod
    def load_from_file(fn):
        try:
            if not ".json" in fn:
                fn += ".json"
            with open(fn,"r") as f:
                data = json.loads(f.read())
            return NeuralNetwork.load_from_weights(data)
        except Exception as e:
            raise e


class Trainer:
    def __init__(self,nn):
        self.nn = nn
        self.trainfit = []
    def train(self,xdata,ydata,epochs=1,learning_rate=0.1):
        with tqdm(total=epochs*len(xdata),unit="epochs") as t:
            for i in range(epochs):
                self._train(xdata,ydata,epochs,learning_rate,pb=t)
    def _train(self,xdata,ydata,epochs=1,learning_rate=0.1,*args,**kwargs):
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
                j = len(self.nn.W)-j-1
                A = self.nn.A[j]
                if self.nn.biased and j<len(self.nn.W)-1:
                    A = matrixy(add_after(A,1))
                    w = w[:-1,:]
                dw = A.T@d
                self.dW = add_before(self.dW,dw)
                if j > 0:
                    act = self.nn.activation(self.nn.Z[j-1],deriv=True)
                    d = d@w.T
                    d = d*act
                    self.deltas = add_before(self.deltas,d)
            if "pb" in kwargs:
                kwargs["pb"].update(1)
            self.update_weights(learning_rate)
            err = 0.5*(np.array(Y)-np.array(yh[0]))**2
            trains += np.linalg.norm(err)
        t = trains
        self.trainfit.append(t)
    def update_weights(self,learning_rate=0.1):
        for i,w in enumerate(self.nn.W):
            self.nn.W[i] = np.subtract(w,self.dW[i]*learning_rate)
def main():
    X = [0,1,1,0,0]
    Y = [1,1]
    nn = NeuralNetwork([5,4,3,2])
    Yh = nn.forward(X)[0]
    trainer = Trainer(nn)
    trainer.train([X],[Y])
    print("Yh:",Yh)
    print("Yh after train:",nn.forward(X)[0])

if __name__=="__main__":
    main()