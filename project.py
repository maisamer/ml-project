import numpy as np
import pandas as pd
import operator
import math

def Sigmoid(z):
	re = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return re

def Hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in xrange(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J



def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	for j in xrange(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta
   
def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in xrange(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta

def predict(test_set,theta) :
    rslt = []
    for i in test_set :
        prediction = round(Hypothesis(i,theta))
        rslt.append(prediction)   
    return rslt
        
def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x)):
        d += pow((float(x[i])-float(xi[i])),2)  #euclidean distance
    d = math.sqrt(d)
    return d

def accuracy(goal_test,goal_predict):
    correct = 0
    for i,j in zip(goal_test,goal_predict) :  
        if i == j:
            correct += 1
    accuracy = float(correct)/len(goal_test)  #accuracy 
    return accuracy

#KNN prediction and model training
def knn_predict(test_data , train_data,goal_train , k_value) :
    result = []
    for i in test_data:
        eu_Distance =[]
        knn = []
        my_dict={}
        for j,l in zip(train_data,goal_train) :
            eu_dist = euclideanDist(i, j)
            eu_Distance.append((l, eu_dist))
            
        eu_Distance.sort(key=operator.itemgetter(1))
        
        knn = eu_Distance[:k_value]
        #print(knn)
        for k in knn:
            my_dict[k[0]] = my_dict.get(k[0] , 0) + 1
     
        mitem = max(my_dict.iteritems(), key=operator.itemgetter(1));
        result.append(mitem[0])
            
    return result;




dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:,2:-1].values
goal = dataset.iloc[:,-1].values

# standrize the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# split into train and test
from sklearn.cross_validation import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,train_size =0.8,random_state=0)

# knn algorithm
knn = knn_predict(test_set,train_set,goal_train,5);
print ("knn algorithm accuracy is : %f" %(accuracy(goal_test,knn)))
# Logistic_Regression
theta=[0,0]
iterations = 1500
alpha = 0.01
Logistic_Regression(train_set, goal_train,  alpha,theta, iterations)
Logistic_Regression = predict(test_set,theta)
print("Logistic_Regression algorithm accuracy is : %f" %(accuracy(goal_test,Logistic_Regression)))
















