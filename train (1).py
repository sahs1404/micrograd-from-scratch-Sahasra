from value import Value
from nn import MLP

# ----- 1. Create a tiny dataset -----
# We will learn XOR-like pattern
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]

ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]  # targets


# ----- 2. Build a model -----
model = MLP(2, [4, 1])  # input 2 → hidden 4 → output 1

# ----- 3. Training -----
learning_rate = 0.1

for epoch in range(100):
    # forward pass
    ypred = [model(x) for x in xs]

    # compute squared error loss
    loss = sum((yp - yt)*(yp - yt) for yp, yt in zip(ypred, ys))

    # reset gradients
    for p in model.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # gradient descent step
    for p in model.parameters():
        p.data += -learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss = {loss.data}")

print("\nFinal predictions:")
for x in xs:
    print(model(x).data)