import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
mysigma = 0.024
myC = 20

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=mysigma):
    return np.exp(-((linalg.norm(y - x))**2)*mysigma)#/ (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.kernel == linear_kernel:#self.w != None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv in zip(self.a,self.sv):
                    s += a* self.kernel(X[i], sv)
                    #s += a*sv_y*self.kernel(X[i], sv)
                #print(s,i)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    
    def save(self):
        np.save('paramater/self_w.npy', self.w)
        np.save('paramater/self_b.npy', self.b)
        np.save('paramater/self_sv_y.npy', self.sv_y)
        np.save('paramater/self_sv.npy', self.sv)
        np.save('paramater/self_a.npy', self.a)

    def load(self):
        self.w = np.load('paramater/self_w.npy')
        self.b = np.load('paramater/self_b.npy')
        self.sv_y = np.load('paramater/self_sv_y.npy')
        self.sv = np.load('paramater/self_sv.npy')
        self.a = np.load('paramater/self_a.npy')


if __name__ == "__main__":
    #读取训练数据
    train_data = np.load('data/sentence2vec_data_2.npy')
    #train_data = (train_data - train_data.mean())/train_data.std()	
    train_label = np.load('data/classification_train_Label_2.npy')
    train_label = train_label.astype('int32')
    for i in range(len(train_label)): # 修改训练数据集标签
        if train_label[i] == 0:
            train_label[i] = -1
    #划分数据集
    test_data = train_data[23500:24000]
    test_label = train_label[23500:24000]
    #print(test_label)
    #train_data = train_data[0:8000]
    #train_label = train_label[0:8000]
    
    #创建svm学习器
    clf = SVM(kernel= gaussian_kernel, C=myC)
    #clf.fit(train_data,train_label)
    clf.load()#加载参数
    print(mysigma,'|', myC)
    print('-------saving------')
    y_predict = clf.predict(test_data)
    print(y_predict,test_label)
    correct = np.sum(y_predict == test_label)
    print(correct/len(test_data))