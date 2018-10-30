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

class PyNeural:
    def __init__(self,*args,**kwargs):
        arg = self._parse_args(*args,**kwargs)
        self.size = arg["size"]
        self.is_input = arg["is_input"]
        self.parent = None
        if not arg["is_input"]:
            self.parent = arg["parent"]
            self.biased = arg["biased"]
            self._init_weights()
            self.activation = self.choose_activation(arg["activation"])
    def _forward(self,X):
        self.X = matrixy(X)
        if self.is_input:
            self.A = X
            return X
        else:    
            if self.biased:
                X = add_after(X,1)
            X = matrixy(X)
            self.Z = X@self.W
            self.A = self.activation(self.Z)
            return self.A
    def choose_activation(self,activation):
        if type(activation)==str:
            if activation=="sigmoid":
                return sigmoid
                self.act_name = activation
        else:
            return activation
    def _init_weights(self):
        dim = [self.parent.size, self.size]
        if self.biased:
            dim[0] += 1
        self.W = 2*np.random.rand(*dim)-1
    def _parse_args(self,*args,**kwargs):
        options = ["parent", "size", "is_input", "biased"]
        arg = {}
        c = 0
        if not isinstance(args[0], PyNeural):
            c += 1
        for i,a in enumerate(args):
            if i+c<len(options):
                arg[options[i+c]] = args[i]
            else:
                raise ValueError("Too much arguments passed")
        for key in kwargs:
            arg[key] = kwargs[key]

        required = {"activation":"sigmoid", "parent":None, "biased":True, "is_input":False}
        for e in required:
            if not e in arg:
                arg[e] = required[e]

        return arg
    def predict_one(self,X):
        X = matrixy(X)
        E = self
        llist = []
        while E != None:
            llist.append(E)
            E = E.parent
        llist = llist[::-1]
        for layer in llist:
            X = layer._forward(X)
        return X

    def predict(self,X):
        Y = []
        for x in X:
            Y.append(self.predict_one(x))
        return Y
    def calc_delta(self,Y,output=False):
        if output:
            Y = matrixy(Y)
            sp = self.activation(self.Z,deriv=True)
            self.D = -(Y-self.A)*sp
        elif self.is_input:
            return
        else:
            # then the Y needs to be the next layer in forward chain
            sp = self.activation(self.Z,deriv=True)
            if self.biased:
                self.D = Y.D@Y.W[:-1,:].T*sp
            else:
                self.D = Y.D@Y.W.T*sp
    def calc_dW(self):
        if self.is_input:
            return
        A = self.X
        if self.biased:
            A = matrixy(add_after(A,1))
        self.dW = A.T@self.D
    def optimize(self,learning_rate=0.01):
        if not self.is_input:
            self.W = self.W-self.D*learning_rate
    def fit_one(self,x,y,learning_rate=0.01):
        llist = []
        E = self
        while E!=None:
            llist.append(E)
            E = E.parent
        isinput = True
        for i,layer in enumerate(llist):
            G = y
            if not isinput:
                G = llist[i-1]
            layer.calc_delta(G,isinput)
            layer.calc_dW()
            layer.optimize()
            isinput = False
    def fit(self,X,Y,n_epochs=1,learning_rate=0.01):
        for i in range(n_epochs):
            for x,y in zip(X,Y):
                self.fit_one(x,y,learning_rate)
            print("Epoch:",i+1)
    def __str__(self):
        lsize = []
        E = self
        while E != None:
            lsize.append(E.size)
            E = E.parent
        return "<PyNeuro({})>".format(lsize[::-1])
    def __repr__(self):
        return self.__str__()

def input_data(ninput):
    ilayer = PyNeural(ninput,True)
    return ilayer
def fully_connected(nn,lsize,activation="sigmoid"):
    lay = PyNeural(nn,lsize,False,True,activation="sigmoid")
    return lay

def main():
    X = [0,1,0,1,1]
    Y = [1,1]
    nn = input_data(5)
    nn = fully_connected(nn,4)
    nn = fully_connected(nn,3)
    nn = fully_connected(nn,2)
    Yh = nn.predict_one(X)
    print(nn)
    nn.fit([X],[Y],100)
    print("Yh:",Yh)
    Yh2 = nn.predict_one(X)
    print("Yh2:",Yh2)
if __name__=="__main__":
    main()