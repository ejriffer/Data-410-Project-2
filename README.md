# Data-410-Project-2

## Locally Weighted Regression
Locally Weighted Regression (Lowess) is a non-parametric linear regression in 'local' segments to get an overall curve made up of smaller straight lines. This combines the simplicity of a linear regression with the flexibility of nonlinear regressions. 

LWR uses Euclidean Distance to 'weight' different data points based on their distance to the data points that are closer to the one we are trying to predict. The weights are determined by a chosen kernel function. The different kernels weight slightly differently but in general the closer a data point is to the point we are trying to predict the heavier the weight will be.

The below code shows the functions for a tricubic kernel and the lowess_reg function which executes a lowess. The hyperperameter 'tau' that can be changed in order to improve accuracy.

```
# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
 
# Lowess Code
def lowess_reg(x, y, xnew, kern, tau):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # IMPORTANT: we expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
```
The code below uses the cars.csv file to show the lowess_reg function in action.  
```
x = data['WGT'].values
y = data['MPG'].values

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))

yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scaled,tricubic,0.1)

mse(yhat_test,ytest)
```
The mse (mean squared error) of the predicted vs. actual y values are a good indicator of a regressiors accuracy. For the lowess code seen above the mse was ~15.96.

Below is the code to obtain a graph of xtest_scaled and y_test and the regression that we obtained through Lowess. 

 ```
 M = np.column_stack([xtest_scaled,yhat_test])
 M = M[np.argsort(M[:,0])]
 
 plt.scatter(xtest_scaled,ytest,color='blue',alpha=0.5)
 plt.plot(M[:,0],M[:,1],color='red',lw=2)
 ```
 
 <img width="374" alt="Screen Shot 2022-02-11 at 4 32 14 PM" src="https://user-images.githubusercontent.com/74326062/153673216-4384f17e-0baa-4b00-98a2-7a988bd44504.png">

## Random Forest Regression
Another type of regression is Random Forest Regression (RFR). RFR is an ensemble learning technique that builds multiple decision trees, making a 'forest'. RFR is more resilient to outliers, which can imporve external validity between the train and test sets. Another benefit is that the python library sklearn has an RFR function built in, so no code needs to be written. 

Two important hyperperameters for RFR in the sklearn library are n_estimators and max_depth. n_estimators is the number of trees in the forest which has a default value of 100 and max_depth is the maximum depth of each tree which defaults to splitting until all leaves are 'pure' or each leaf has one value. 

The below code shows RFR in action with the same 'cars.csv' file as in the LWR example.

```
x = data['WGT'].values
y = data['MPG'].values

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))

rf = RandomForestRegressor(n_estimators=100,max_depth=3)
rf.fit(xtrain_scaled,ytrain)

mse(ytest,rf.predict(xtest_scaled))
```
 For the RFR code seen above the mse was ~15.93.
 
 Using the same code as seen above here is the regression that we obtained through RFR. 
 
 <img width="374" alt="Screen Shot 2022-02-11 at 4 33 57 PM" src="https://user-images.githubusercontent.com/74326062/153673388-b35f6ad7-f635-40a7-9d72-4c04fe1da423.png">

## Comparison

In order to accuratley compare Lowess and RFR you would need to split the data into train and test groups, as seen above and use a KFold validation technique to make sure your number are accurate, and not due to chance if the data splits in a certain way. 

The below code uses the same data as above and compares Lowess with RFR.

```
# initiate KFold
kf = KFold(n_splits=10,shuffle=True,random_state=310)

# lists to catch the mse of each lowess and RFR regression
mse_lwr = []
mse_rf = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_lwr = lowess_reg(xtrain.ravel(),ytrain,xtest.ravel(),tricubic,0.1)
  rf = RandomForestRegressor(n_estimators=100,max_depth=4)
  rf.fit(xtrain,ytrain)
  yhat_rf = rf.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_rf.append(mse(ytest,yhat_rf))
  
print('the mse for rf is:' + str(np.mean(mse_rf)))
print('the mse for lwr is:' + str(np.mean(mse_lwr)))
```

With the above hyperperameters we see that the mse for RFR is ~19.39 and the mse for Lowess is ~18.19. So, in this case we can determine that Lowess is a slightly superior model to RFR. 
