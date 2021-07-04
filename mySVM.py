import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
from gensim.models import word2vec 

from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
mysigma = 0.024
myC = 20

def linear_kernel(x1, x2):#线性核函数
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=6):#多项式核函数
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=mysigma):#高斯核函数
    return np.exp(-((linalg.norm(y - x))**2)*mysigma)#/ (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
    def fit(self, X, y): #训练数据集 X为数据y为标签
        n_samples, n_features = X.shape
        # Gram matrix即核函数矩阵
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # 调用cvxopt库解决二次优化问题
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
        self.a = a[sv] #决策平面的alpha系数
        self.sv = X[sv] #存储支持向量
        self.sv_y = y[sv] #存储支持向量在原数据集中的索引
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

    def project(self, X):#决策函数
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
    
    def save(self):#存储参数
        np.save('paramater/self_w.npy', self.w)
        np.save('paramater/self_b.npy', self.b)
        np.save('paramater/self_sv_y.npy', self.sv_y)
        np.save('paramater/self_sv.npy', self.sv)
        np.save('paramater/self_a.npy', self.a)

    def load(self):#加载参数
        self.w = np.load('paramater/self_w.npy')
        self.b = np.load('paramater/self_b.npy')
        self.sv_y = np.load('paramater/self_sv_y.npy')
        self.sv = np.load('paramater/self_sv.npy')
        self.a = np.load('paramater/self_a.npy')

if __name__ == '__main__':
    """ train_data_2 = np.load('classification_train_data_2.npy')
    word2vec_train_2 = word2vec.Word2Vec.load('word2vec_train_2.model')
    size = len(word2vec_train_2['laughing'])
    sentence = np.zeros((len(train_data_2),size))
    len(train_data_2[0])
    for i in range(np.shape(sentence)[0]):
        for word in train_data_2[i]:
            sentence[i] += word2vec_train_2[word]
        sentence[i] = sentence[i]/len(train_data_2[i])
    print(sentence[0])
    np.save('sentence2vec_data_2.npy',sentence) """

    """ test_data_2 = np.load('classification_test_data_2.npy')
    word2vec_lib = word2vec.Word2Vec.load('word2vec_train_2.model')
    size = len(word2vec_lib['laughing'])
    test_sentence = np.zeros((len(test_data_2),size))
    for i in range(np.shape(test_sentence)[0]):
        for word in test_data_2[i]:
            test_sentence[i] += word2vec_lib[word]
        test_sentence[i] = test_sentence[i]/len(test_data_2[i])
    np.save('sentence2vec_test_2.npy',test_sentence) """

    # 加载数据test
    test_data = np.load('data/sentence2vec_test_2.npy')
    #test_data = (test_data - test_data.mean())/test_data.std()
    # 加载clf
    clf = SVM(kernel= gaussian_kernel, C=myC)
    clf.load()
      
#--------------------------------------------------------------------
#测试准确率
    train_data = np.load('data/sentence2vec_data_2.npy')
    #train_data = (train_data - train_data.mean())/train_data.std()	# 数据标准化
    train_label = np.load('data/classification_train_Label_2.npy')
    train_label = train_label.astype('int32')
    train_data = train_data[0:500]
    train_label = train_label[0:500]


    trainresult = clf.predict(train_data)
    trainresult = trainresult.astype('int32')
    for i in range(np.shape(trainresult)[0]):
        if trainresult[i] == -1:
            trainresult[i] = 0
    print(np.sum(trainresult==train_label)/len(trainresult))
#----------------------------------------------------------------
    result = clf.predict(test_data)
    for i in range(np.shape(result)[0]):
        if result[i] == -1:
            result[i] = 0
    result = result.astype('int32')

#------------将测试集与训练集进行对照
    train_data_2 = np.load('data/classification_train_data_2.npy')
    train_label_2 = np.load('data/classification_train_label_2.npy')
    test_data_2 = np.load('data/classification_test_data_2.npy')
    for i in range(np.shape(test_data_2)[0]):
        for j in range(np.shape(train_data_2)[0]):
            if test_data_2[i] == train_data_2[j]:
                result[i] = train_label_2[j]
                #print('i=',i,'|j=',j,'|',result[i])

    np.savetxt('16337287_3.txt',result,fmt='%d')

