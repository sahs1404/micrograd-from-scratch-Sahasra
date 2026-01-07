import math

class Value:
    """
    A scalar value that supports automatic differentiation.
    Stores:
    - data: the numerical value
    - grad: the gradient (dOutput/dThis)
    - _backward: function applied in backprop
    - _prev: child nodes in computation graph
    - _op: operation used to create it (for debugging)
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # --------- BASIC OPERATORS ---------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')


        def _backward():
            # d(out)/d(self) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule:
            # d(a*b)/da = b
            # d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
        
    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)


    # --------- ACTIVATION FUNCTIONS ---------

    def tanh(self):
        x = self.data
        # tanh(x) = (e^2x - 1) / (e^2x + 1)
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # derivative of tanh: 1 - tanh(x)^2
            self.grad += (1 - t*t) * out.grad
        out._backward = _backward
        return out

    # --------- BACKPROP ---------

    def backward(self):
        """
        Runs backpropagation starting from this value.
        Must be called on the final scalar output.
        """
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)         # topological ordering
        self.grad = 1.0     # Seed gradient

        for node in reversed(topo):
            node._backward()
