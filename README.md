# Data-410-Project-2

## Locally Weighted Regression
Locally Weighted Regression (LWR) is a non-parametric linear regression in 'local' segments to get an overall curve made up of smaller straight lines. This combines the simplicity of a linear regression with the flexibility of nonlinear regressions. 

LWR uses Euclidean Distance to 'weight' different data points based on their distance to the data points that are closer to the one we are trying to predict. The weights are determined by a chosen kernel function. The different kernels weight slightly differently but in general the closer a data point is to the point we are trying to predict the heavier the weight will be.

The below code shows the functions for a tricubic kernel and the lowess_reg function which executes a LWR. The hyperperameter 'tau' that can be changed in order to improve accuracy.

```
# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
 
# LWR Code
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
The mse (mean squared error) of the predicted vs. actual y values are a good indicator of a regressiors accuracy. For the LWR code seen above the mse was ~15.96.

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
