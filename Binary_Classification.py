
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt


np.random.seed(12)
observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], observations)

X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(observations), np.ones(observations)))

plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1],
            c = Y, alpha = .4)


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def cost_fun(features, target, Theta):
    scores = np.dot(features, Theta)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(features, target, num_steps, learning_rate):
    
    intercept = np.ones((features.shape[0], 1))
    features = np.hstack((intercept, features))
        
    Theta = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, Theta)
        predictions = sigmoid(scores)
        
        error = target - predictions
        
        gradient = np.dot(features.T,error)
        Theta += learning_rate * gradient

        
        if step % 10000 == 0:
            print (cost_fun(features, target,Theta))
        
    return Theta

Theta = logistic_regression(X, Y,
                     num_steps = 50000, learning_rate = .00005)

print ("Thetas",Theta)


final_scores = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), Theta)
preds = np.round(sigmoid(final_scores))
print ('Accuracy: {0}'.format((preds == Y).sum().astype(float) / len(preds)) )

plt.figure(figsize = (12, 8))
plt.scatter(X[:, 0], X[:, 1],
            c = preds == Y - 1, alpha = .8, s = 30)

