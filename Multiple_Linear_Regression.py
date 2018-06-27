# Multiple Linear Regression
'''
Just like in the previous dataset we explored Linear Regression in 1 variable in which the years of experience was the only dependent variable.
This particular dataset contains the details of 50 startup's and predicts the profit of a new Startup based on certain features.
To Venture Capitalists this could be a boon as to whether they should invest in a particular Startup or not.
So lets say that you work for a Venture Capitalist and your firm has hired you as a Data Scientist to derive insights into the data,
and help them to predict whether a particular startup would be safe to invest in or not.
We can also derive useful insights into the data by actually seeing as to what difference does it make if a Startup is launched in a particular state.
Or Which startup's end up performing better by seeing that if they spent more money on marketing or was it their stellar R&D department which led them to this huge profit.

'''
'''
What I am about to mention might be more on the Theoretical side of Machine Learning but still a bit of theoretical knowledge also doesnt hurt! :P
Before proceeding with building a Linear Regression model there are certain assumptions which should be true 
1)LINEARITY
2)HOMOSCEDASTICITY
3)MULTIVARIATE NORMALITY
4)INDEPENDENCE OF ERRORS
5)LACK OF MULTICOLLINEARITY
Well I wont be discussing all these things, but If we are to proceed with Linear Regression as our ML algorithm in the Future...We must try and keep a track of them.
Those Interested can research about them themselves!! 
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
#The dataset contains the following features(independent variables):--
#........1)R&D Spend-Total amount of money spent on Research and Development by the startup.
#........2)Administration-Total amount of money spent on Administration by the startup.
#........3)Marketing Spend-Total amount of money spent on Marketing by the startup.
#........4)State-The State or region in which the startup is launched or operates.
X = dataset.iloc[:, :-1].values
#The dependent variable Profit which tells you the Profit acquired by the Startup.
y = dataset.iloc[:, 4].values

"""
If we take a look at our Dataset we can clearly see that State is a String type variable and like we have discussed,
We cannot feed String type variables into our Machine Learning model as it can only work with numbers.
To overcome this problem we use the Label Encoder object and create Dummy Variables using the OneHotEncoder object...
Lets say that if we had only 2 states New York and California namely in our dataset then our OneHotEncoder will be of 2 columns only...
Similarly for n different states it would have n columns and each state would be represented by a series of 0s and 1s wherein all columns would be 0
except for the column for that particular state.
For ex:-
If A,B,C are 3 states then A=100,B=010,C=001
I think now you might be getting my point as to how the OneHotEncoder works... 
"""
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#The Linear Regression equation would look like-----> y=b(0)+b(1)x(1)+b(2)x(2)+b(3)x(3)+b(4)D(1)+b(5)D(2)+b(6)D(3)...b(n+3)D(m-1)
#Here D(1)...D(m-1) are the m dummy variable's which we had defined earlier in LabelEncoder and OneHotEncoder
#Well if you are sharp enough you might have noticed that the even though there are m dummy variables we have excluded the last dummy variable D(m)
#The reason to that is a concept called Dummy Variable Trap in Machine Learning...and to avoid that we must always exclude the last Dummy Variable
#If you are more interested then feel free to research a bit on Dummy Variable Trap!!
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
#The Linear Regression equation would look like-----> y=b(0)+b(1)x(1)+b(2)x(2)+b(3)x(3)+b(4)D(1)+b(5)D(2)+b(6)D(3)...b(n+3)D(m-1)
#Importing the Linear Regression Class
from sklearn.linear_model import LinearRegression
#Creating an object of the Linear Regression Class
regressor = LinearRegression()
#Fit the created object to our training set
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Printing the predicted values
print(y_pred)
#To see the difference b/w predicted and actual results we will also print the test values
print(y_test)
#Do you think that its actually the optimal model which we have just built?
#Our model contains all the features some of which are statistically insignificant to our predictions.
#What we need to do is find a team of all independent variables which are infact helpful to our predictions.
#For that we use another method called Backward Elimination in order to build an optimal model...
#But that's another story and would require an Article of its own :P So stay tuned!!