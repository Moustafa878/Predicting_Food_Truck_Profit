import matplotlib.pyplot as plt
import numpy as np
#======================= Cost Function =======================
def  compute_cost(x,y,theta):
    m=len(y)
    prediction=np.dot(x,theta)
    j=(sum(np.square(prediction-y)))/(2*m)
    return j
#======================= Gradient Descent =======================
def compute_gradient (x,y,theta,num_of_iter,alpha):
    m = len(y)
    j_history = np.zeros(num_of_iter)
    for i in range(0,num_of_iter):
      prediction = np.dot(x, theta)
      theta= theta- (alpha / m) * np.dot(x.transpose(),(prediction - y))
      j_history[i] =compute_cost(x,y,theta)
    return (theta,j_history)



#======================= load and Plot data =======================
data=np.loadtxt('ex1data1.txt',delimiter=',')
X=np.c_[data[:,0]]
y=np.c_[data[:,1]]

plt.scatter(X,y,s=30,c='red',marker='x',linewidths=1)
plt.ylim(-5,25)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

#=======================Test cost function =======================
X=np.insert(X,0,np.ones(X.shape[0]),axis=1)
theta=np.zeros((2,1))
j=compute_cost(X,y,theta)
print('cost function when theta=zeros : ', j)
j=compute_cost(X,y,np.matrix([[-1],[2]]))
print('cost function when theta=[[-1],[2]] : ', j)
#=======================Test Gradient Descent  =======================
iterations = 1500
alpha = 0.01
theta , Cost_J =compute_gradient(X,y,theta,iterations,alpha)
print('optimal theta after running Gradient Descent : ',theta)

# ploting cost function with number of iteration after running gradient descent

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()
#Test
predict1 = np.dot([[1, 3.5]],theta)
print('For population = 35,000, we predict a profit of : ',predict1*10000)
predict2 =np.dot([[1, 7]], theta)
print('For population = 70,000, we predict a profit of : ',predict2*10000)
#=============  Visualizing J(theta_0, theta_1) =============
xx = np.arange(5,24)
yy = theta[0]+theta[1]*xx

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')
plt.ylim(-5,25)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show()
