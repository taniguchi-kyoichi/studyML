import numpy as np
import matplotlib.pylab as plt



def step_function(x):
    return np.array(x > 0, dtype=np.int)


# -- 活性化関数 --
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def identify_function(x):
    return x

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# -- 損失関数 --

def mean_square_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# -- 勾配法 --

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    #it = np.nditer(x, flags=['multi_index'], open_flags=['readwrite'])
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext() 

    return grad

        


def gradient_decent(f, init_x, lr=0.01, step_num=100):
    x = init_x 

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
    
def function_2(x):
    return x[0]**2 + x[1]**2



def main():
    init_x = np.array([-3.0, 4.0])
    res = gradient_decent(function_2, init_x=init_x, lr=1e-10, step_num=100)
    print(res)

