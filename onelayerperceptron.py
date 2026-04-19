import numpy as np

class OneLayerPerceptron:
    def __init__(self, lr): 
        self.lr = lr  #Скорость обучения

    def activation(self,x):  #Функция активации
        return 1 if x>=0 else 0
    
    
    def train(self,X,y,epoch):
        self.weight = np.random.rand(X.shape[1])
        self.b = np.random.rand(1)

        for i in range(epoch):
            for j in range(len(X)):
                train_predict = self.activation(np.dot(X[j], self.weight)+self.b)

                error = y[j] - train_predict  #Функция потерь
                self.weight+=self.lr*error*X[j]
                self.b+=self.lr*error
    
    def predict(self, inp):
        result = []

        for x in inp:
            y = self.activation(np.dot(x, self.weight)+self.b)
            result.append(y)

        return print(f'Вход: {inp}, выход: {result}')

    
