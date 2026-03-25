import numpy as np

lr = 0.1 #learning rate
bias = np.random.rand(1)
weight = np.random.rand(2)

ins_value = np.array([[1,1], [1,0], [0,0]])
pred_value = np.array([1,0,1])

def activation(x):
    return 1 if x >= 0 else 0

for epoch in range(100):
    for i in range(len(ins_value)):
        layer = np.dot(ins_value[i], weight) + bias
        predict = activation(layer)

        error = pred_value[i] - predict
        weight += error*lr*ins_value[i]
        bias += error*lr

print(f"Обученные веса: {weight}, смещение: {bias}")
for xi in ins_value:
    res = 1 if (np.dot(xi, weight) + bias) >= 0 else 0
    print(f"Вход {xi} -> Ответ {res}")