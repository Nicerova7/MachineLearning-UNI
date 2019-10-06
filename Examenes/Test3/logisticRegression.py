import numpy as np

class LogisticRegression:
    """ My implementation of Logistic Regression with l2 regularization parameter """
    
    def __init__(self, regularization = 0, max_iter = 2000):
        self.regularization = regularization
        self.max_iter = max_iter
        
    def fit(self, X, y):
        
        def logitprobs(x,theta):
            return 1/(1 + np.exp(-np.dot(x, theta)))

        def findW(pi):
            W = pi*(1-pi)
            return W
        
        def g(theta):
            grad = np.dot(X_new.transpose(), logitprobs(X_new,theta) - y) + self.regularization*theta
            grad[0] -= self.regularization*theta[0]
            return grad
        def H(theta):
            pi = logitprobs(X_new,theta)
            W = findW(pi)
            hessian = np.dot(X_new.transpose()*(W), X_new) + I*self.regularization
            hessian[0,0] -= self.regularization        
            return hessian
        
        bias = np.ones(len(X))
        X_new = np.column_stack((bias, X))
        I = np.eye(X_new.shape[1])
        theta = np.zeros(X_new.shape[1])
        iter_i = 10000
        i = 0
        
        while iter_i > 1e-5 and i < self.max_iter:
            
            root_dif = np.linalg.solve(H(theta), -g(theta))
            theta = theta + root_dif
            iter_i = np.sqrt(np.dot(root_dif,root_dif))
            i += 1
            
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return theta
    
    def predict_proba(self, X):
        mu = 1/(1 + np.exp(-(self.intercept_ + np.dot(X, self.coef_))))
        return np.column_stack((1-mu,mu))
    def predict(self, X):
        return np.round(self.predict_proba(X)[:,1])
    def score(self, X, y):
        return sum(self.predict(X) == y)/len(y)