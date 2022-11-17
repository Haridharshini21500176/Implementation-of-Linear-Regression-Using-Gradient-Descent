# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the needed packages
2.Read the txt file using read_csv
3.Use numpy to find theta,x,y values
4.To visualize the data use plt.plot


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.Haridharshini
RegisterNumber:  212221230033
*/
```
```
#import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("ex1.txt",header=None)

plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Take in a np array X,y,theta and generate the cost function of using theta as parameter in a linear regression model
"""
def computeCost(X,y,theta):
    m=len(y) #length of the training data
    h=X.dot(theta) #hypothesis
    square_err=(h-y)**2
    
    return 1/(2*m)*np.sum(square_err) #returning J

df_n=df.values
m=df_n[:,0].size
X=np.append(np.ones((m,1)),df_n[:,0].reshape(m,1),axis=1)
y=df_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta) #call the function

"""
Take in np array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
return theta and the list of the cost of theta during each iteration
"""
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len👍
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(),(predictions -y))
        descent = alpha*(1/m )*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Takes in numpy array of x and theta and return the predicted value of y based on theta
"""
def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![193451349-54044b08-c80f-4f05-ad97-141dc786bb5a](https://user-images.githubusercontent.com/94168395/202348989-b39c1350-260c-40c9-99ae-287853761842.png)

![193451383-381c1f04-a881-4799-8aff-51d80eebe419](https://user-images.githubusercontent.com/94168395/202349039-25efdf56-fc01-474a-bcce-73dee02430ad.png)

![193451415-e3355674-f4cb-475c-ace4-0865cc3ed0c9](https://user-images.githubusercontent.com/94168395/202349068-860f06ea-22ef-4183-af0a-3d88411d0081.png)
![193451428-b9cb0d87-177a-4765-a181-f308f0bb54f6](https://user-images.githubusercontent.com/94168395/202349088-aa56f2cb-dcf6-4fa9-9cf4-7b02449e9130.png)

![193451434-623ee4ac-6eff-40fc-9748-0aaab1648297](https://user-images.githubusercontent.com/94168395/202349126-73283a0b-0752-4880-804c-5b8c16dbf87c.png)

![193451460-c1348099-e723-4361-9d7e-dfd8fbe47f2d](https://user-images.githubusercontent.com/94168395/202349154-6ada240c-9d01-4742-9c54-58ae8a54322e.png)

![193451463-efdafc0d-f7a6-4e15-91c5-daea849643cf](https://user-images.githubusercontent.com/94168395/202349179-ea041267-371a-4fe2-ba8e-7800ac541970.png)

![193451468-d702ab15-e83b-45f2-b30b-aaca4f0508d7](https://user-images.githubusercontent.com/94168395/202349227-0b4f2d67-856e-46c5-ad9f-fc5f763f5982.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
