import numpy as np
import cvxopt

__author__ = "Yacine Sibous"


class LinearKernel(object):

    ''' Implement linear kernel
    '''
    def linear_kernel():
        def lin(X, y):
            return np.inner(X, y)
        return lin


class TrainSVM(object):

    ''' The methods in this class will
        be used to train our SVM
    '''
    def __init__(self, kernel, c):
        self.kernel = LinearKernel.linear_kernel()
        self.c = c

    def gram_matrix(self, X):

        ''' Compute the gram matrix
        '''
        sample_size, feature_size = X.shape
        Z = np.zeros((sample_size, feature_size))

        for j, x_o in enumerate(X):
            for i, y_o in enumerate(X):
                Z[j, i] = self.kernel(x_o, y_o)
        return Z

    def multipliers(self, X, y):

        '''Compute Lagrangian Multipliers
        '''
        sample_size, feature_size = X.shape
        Z = self.gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * Z)
        q = cvxopt.matrix(-1 * np.ones(sample_size))

        G_s = cvxopt.matrix(np.diag(np.ones(sample_size) * -1))
        G_sl = cvxopt.matrix(np.diag(np.ones(sample_size)))
        G = cvxopt.matrix(np.vstack((G_s, G_sl)))

        h_s = cvxopt.matrix(np.zeros(sample_size))
        h_sl = cvxopt.matrix(np.ones(sample_size) * self.c)
        h = cvxopt.matrix(np.vstack((h_s, h_sl)))

        A = cvxopt.matrix(y, (1, sample_size))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def train(self, X, y):

        '''
            Given a set of training features X and training output y,
            we return an SVM predictive model
        '''
        lagrangian = self.multipliers(X, y)
        return self.predict(X, y, lagrangian)

    def predict(self, X, y, lagrangian):
        s_mult = lagrangian[s_v_i]
        s_v = X[s_v_i]
        s_v_l = y[s_v_i]

        bias = np.mean([y_i - PredictSVM(bias=0.0, weights=s_mult, s_vectors=s_v, s_v_l=s_v_l).predict(x_i) for (y_i,x_i) in zip(s_v_l,s_v)])
        return PredictSVM(bias=bias, weights=s_mult, s_vectors=s_v, s_v_l=s_v_l, kernel=self.kernel)


class PredictSVM(object):

    ''' Using our training model we could now predict
        resuts for new input data
    '''

    def __init__(self, weights, bias, s_vectors, s_v_l, kernel):

        self.weights = weights
        self.bias = bias
        self.s_vectors = s_vectors
        self.s_v_l = s_v_l
        self.kernel = kernel

    def predictSVM(self, X):

        ''' Return SVM prediction given a test data set X
        '''

        final = self.bias

        for x_i, y_i, z_i in zip(self.weights, self.s_vectors, self.s_v_l):

            final += self.kernel(y_i, X) * x_i * z_i

        return np.sign(final).item()
