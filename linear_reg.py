import numpy as np
import matplotlib.pyplot as plt

def gen_simple_data():
	D = 2 #dimensions
	s = 2 # separation

	mu1  = np.array([0,0])
	mu2  = np.array([s,s])

	N = 1000 #number of points
	X = np.zeros((N, D))
	X[:500, :] = np.random.randn(500, D) + mu1
	X[500:, :] = np.random.randn(500, D) + mu2

	Y = np.zeros(1000)
	Y[500:] = 1

	return X,Y

def gen_noisy_trend():
	
	N = 1000
	eta = 0.1 * np.random.randn(N)
	X = 0.001 * np.arange(1000)
	Y = X + eta

	return X, Y
    
def predict(w, X):
	yhat = w * X

	return yhat

def train(w, X, Y, n_iter):
	for i in xrange(n_iter):
		w = w - eps * (w*X-Y).dot(X)
	return w

def get_simple_data():
    # assume 3 means
    D = 2 # so we can visualize it more easily
    s = 4 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900 # number of samples
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3
    return X


w = 0
eps = 0.0001
X,Y = gen_noisy_trend()
w1 = train(w, X, Y, 10)
yhat1 = predict(w1, X)
w2 = train(w, X, Y, 100)
yhat2 = predict(w2, X)
w3 = train(w, X, Y, 300)
yhat3 = predict(w3, X)
plt.plot(X, Y)
plt.plot(X, yhat1)
plt.plot(X, yhat2)
plt.plot(X, yhat3)
plt.show()

print "final ws are:", w1, w2, w3