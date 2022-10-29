from dataset.mnist import load_mnist
import functions


from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import pickle

"""
flatten: 入力を１次元にする
normalize: 入力を正規化する
one_hot_label: one_hot表現とする（正解だけ1、それ以外は0の配列）
"""
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def main():
    x, t = get_data()
    network = init_network()
    # バッチ処理を実装
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

    


def img_show_test():
    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = functions.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = functions.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = functions.softmax(a3)

    return y






main()