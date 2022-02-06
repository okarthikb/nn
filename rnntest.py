from nn import *


N = 1000
rnn = RNN(3, 3, 1, 4, 4, 4, [4], [sigmoid], sigmoid, sigmoid)

xs = genwbs(rnn.xn, rnn.xdim)
ys = genwbs(rnn.yn, rnn.ydim)
loss = []

for _ in range(N):
    p = rnn(xs)
    loss.append(((p - ys) ** 2).sum())
    dw, db = rnn.rnnbackward(p - ys)
    rnn.w -= 1e-1 * dw
    rnn.b -= 1e-1 * db

print(f"xs\n{xs}\n\nys\n{ys}\n\npred\n{p}\n\nloss\n{loss[-1]}")

fig = plt.figure(figsize=(8, 8))
plt.xlabel("epoch")
plt.ylabel("loss")
plot = plt.plot(list(range(N)), loss)
plt.savefig("loss.png")
