import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self,num_steps=50000, learning_rate= 5e-5, add_intercept=True):
        self.num_steps= num_steps
        self.learning_rate = learning_rate
        self.add_intercept = add_intercept
    def sigmoid(self, z):
        sig = 1/(1+np.exp(-z))
        return sig

    def log_likelihood(self,features,target, weights):
        scores = np.dot(features, weights)
        l1= np.sum(target * scores - np.log( 1 + np.exp(scores)))
        return l1

    def logistic_regression(self,features, target, num_steps, learning_rate, add_intercept = False):
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
            
        weights = np.zeros(features.shape[1])
        
        for step in range(num_steps):
            scores = np.dot(features, weights)
            preds = self.sigmoid(scores)
            error = target - preds
            gradient = np.dot(features.T , error)
            weights = weights + learning_rate * gradient

            
            if step % 10000 == 0:
                print('LogLikelihood',self.log_likelihood(features, target, weights))
            
        return weights
        

    def weights(self,features, target, num_steps = 50000, learning_rate = 5e-5, add_intercept=True):
        weights = self.logistic_regression(features, target,num_steps, learning_rate , add_intercept)
        print('The weights are:',weights)

    

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

logreg = LogisticRegression()
logreg.weights(simulated_separableish_features,simulated_labels)
# finalscores = np.dot(np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
#                                  simulated_separableish_features)), weights)
# preds = np.round(logreg.sigmoid(finalscores))
# accuracy = (preds == simulated_labels).sum().astype(float) / len(preds)

# print('Accuracy is :',accuracy)

# plt.figure(figsize = (12, 8))
# plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
#             c = preds == simulated_labels - 1, alpha = .8, s = 50)
# plt.show()