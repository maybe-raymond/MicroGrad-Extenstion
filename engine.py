import math

class Value:

    def __init__(self, data, _children = (), _op = "", label =""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _op
        self.label = label


    def __repr__(self):
        return f"Value = {self.data}"
    
    def __add__(self, other):
        other = self._standalise_value(other) 
        out = Value(self.data + other.data, (self, other), "+")

        def backpropagation():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward= backpropagation
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = self._standalise_value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def backpropagation():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
    
        out._backward= backpropagation
        return out
    
    def __rsub__(self, other):
        
        return Value(other) - self 

    def __mul__(self, other):
        other = self._standalise_value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def backpropagation():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        
        out._backward = backpropagation
        return out
    
    
    def __truediv__(self, other):
        return self * other**-1
    
    
    def __neg__(self):
        return self *-1
    

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data **other, (self,), f'**{other}')

        def backpropagation():
            self.grad += other * (self.data **(other - 1 )) * out.grad

        out._backward = backpropagation
        return out


    def __rmul__(self, other):
        return self * other
    
    
    def __abs__(self):
        if self.data < 0:
            return self * -1
        return self 
    
    
    def log(self):
        if self.data <= 0:
            raise ValueError(f"Cannot log number {self.data}")
        x = math.log(self.data)

        out = Value(x, (self,), 'log')

        def backpropagation():
            self.grad += (1/self.data) * out.grad

        out._backward = backpropagation
        return out



    
    def tanh(self):
        x = self.data 
        t = (math.exp(x*2) - 1) /  (math.exp(x*2) + 1)
        out = Value(t, (self, ), _op="tanh")


        def backpropagation():
            # 1 - tanh(x) **2
            self.grad += (1 - t **2) * out.grad
                    
        out._backward = backpropagation

        return out
    
    def backward(self):

        self.grad = 1.0

        # get all the nodes
        visted = set()
        nodes = []
         
        def build_list(start):
            if start not in visted:
                visted.add(start)
                for i in start._prev:
                    build_list(i)
                nodes.append(start)
        
        build_list(self)

        for i in reversed(nodes):
            i._backward()

    
    def _standalise_value(self, value):
        # allows us to add raw ints and floats within an expression
        if isinstance(value, (int, float)):
            return Value(value)
        return value

    def relu(self):
        x = self.data 
        r = x if x >= 0 else 0 
        out = Value(r, (self,), _op="relu")

        def backpropagation():
            # RELU derivative:
            # for x > 0 then 1 or 0
            self.grad += (out.data > 0) * out.grad
                    
        out._backward= backpropagation

        return out


    def sigmoid(self):
        x = self.data
        sig = 1 / (1 + math.exp(-x))

        out = Value(sig, (self,) , _op="sigmoid") 

        def backpropagation():
            # Sigmoid derivative:
            # sig(x) * (1 - sig(x)) :: dont't forget the chain rule
            self.grad += (out.data * (1 - out.data)) * out.grad
                    
        out._backward = backpropagation
 
        return out
