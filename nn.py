import numpy as np
import time, json

# activation functions
def sigmoid(x,deriv=False):
    if deriv:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))
def relu(x,deriv=False):

    if deriv:
        return (x>0).astype(int)
    else:
        return  np.maximum(x,0)
def softmax(inputs,deriv=False):
    inputs = inputs[0].copy()
    if -float("inf") in inputs:
        inputs = [0 if i==-float("inf") else i for i in inputs]
        print("Inputs:",inputs)
    inputs = np.array(inputs)
    shifti = inputs-np.max(inputs)
    exps = np.exp(shifti)
    sm = exps/np.sum(exps)
    return matrixy(sm)
def tanh(x,deriv=False):
    if deriv:
        return 1-tanh(x)**2
    else:
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# helper functions
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
        # args can be parent, size, is_input, biased
        # kwargs can be activation, parent, biased or is_input
        arg = self._parse_args(*args,**kwargs)
        self.size = arg["size"]
        self.is_input = arg["is_input"]
        self.parent = None
        if not arg["is_input"]:
            self.parent = arg["parent"]
            self.biased = arg["biased"]
            np.random.seed(int(time.time()))
            self._init_weights() # Init self.W var with random weights
            self.activation = self.choose_activation(arg["activation"])
        
        if self.parent == None:
            self.index = 0
        else:
            self.index = self.parent.index+1

    def save_to_file(self,fn):
        if not fn.endswith(".json"):
            fn += ".json"
        layers = []
        for layer in self.get_layer_list():
            data = {}
            data["size"] = layer.size
            if not layer.is_input:
                data["activation"] = layer.act_name
                data["W"] = json.dumps(layer.W.tolist())
                data["biased"] = layer.biased
            data["is_input"] = layer.is_input
            layers.append(data)
        with open(fn,"w") as f:
            f.write(json.dumps(layers))

    @staticmethod
    def load_from_file(fn):
        if not fn.endswith(".json"):
            fn += ".json"
        with open(fn,"r") as f:
            data = json.loads(f.read())
        for layer in data:
            if layer["is_input"]:
                net = PyNeural(layer["size"],is_input=True)
            else:
                net = PyNeural(net, layer['size'], is_input=False, biased=layer["biased"], activation=layer["activation"])
                W = np.array(json.loads(layer["W"]))
                net.W = W
        return net

    def _forward(self,X):
        self.X = matrixy(X)
        if self.is_input:
            self.A = self.X
            self.Z = self.X
            return self.X
        else:    
            if self.biased:
                X = add_after(X,1)
            X = matrixy(X)
            self.Z = X@self.W
            self.A = self.activation(self.Z)
            return self.A
    def choose_activation(self,activation):
        if type(activation)==str:
            activation = activation.lower()
            if activation=="sigmoid":
                self.act_name = activation
                return sigmoid
            elif activation=="relu":
                self.act_name = activation
                return relu
            # elif activation=="softmax":
            #     self.act_name = activation
            #     return softmax
            elif activation=="tanh":
                self.act_name = activation
                return tanh
        else:
            return activation
    def get_layer_list(self):
        llist = []
        E = self
        while E != None:
            llist.append(E)
            E = E.parent
        return llist[::-1]
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
        if self.is_input:
            return
        elif output:
            Y = matrixy(Y)
            sp = self.activation(self.Z,deriv=True)
            self.E = 0.5*(Y-self.A)**2
            self.D = -(Y-self.A)*sp
        else:
            # then the Y needs to be the next layer in forward chain
            sp = self.activation(self.Z,deriv=True)
            if self.biased:
                # extract the bias weight for delta calculation
                w = Y.W[:-1,:]
                self.D = Y.D@w.T*sp
            else:
                self.D = Y.D@Y.W.T*sp
    def calc_dW(self):
        if self.is_input:
            return
        A = self.parent.A
        if self.biased:
            A = matrixy(add_after(A,1))
        self.dW = A.T@self.D
    def optimize(self,learning_rate=0.1):
        if not self.is_input:
            self.W = self.W-self.dW*learning_rate
    def fit_one(self,x,y,learning_rate=0.1):
        llist = self.get_layer_list()[::-1]
        isinput = True
        Yh = self.predict_one(x)
        for i,layer in enumerate(llist):
            G = y
            if not isinput:
                G = llist[i-1]
            layer.calc_delta(G,isinput)
            layer.calc_dW()
            isinput = False
        for layer in llist:
            layer.optimize()
    def fit(self,X,Y,n_epochs=1,learning_rate=0.1,verbose=False,pb=None):
        if not isinstance(pb,type(None)):
            pb.total = len(X)*n_epochs
        for i in range(n_epochs):
            for x,y in zip(X,Y):
                self.fit_one(x,y,learning_rate)
                if not isinstance(pb,type(None)):
                    pb.update(1)
            if verbose!=False and (i+1)%verbose==0:
                print("Epoch:",i+1)
        if not isinstance(pb,type(None)):
            pb.close()
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
    lay = PyNeural(nn,lsize,False,True,activation=activation)
    return lay

def main():
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
    nn.save_to_file("testing.json")
    Yh2 = nn.predict(X)
    print("Y:",Y)
    print("Y before fit:",[np.round(i[0]).tolist() for i in np.divide(Yh,max(Yh))])
    print("Y after fit :",[np.round(i[0]).tolist() for i in np.divide(Yh2,max(Yh2))])
    print("             ",[i[0].tolist() for i in Yh2])
    print("Done!")
if __name__=="__main__":
    main()