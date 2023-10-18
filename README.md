# ODD2023-Datascience-Ex06
# Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

# Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# Algorithm:
Step1: Read the given Data.

Step2: Clean the Data Set using Data Cleaning Process.

Step3: Apply Feature Transformation techniques to all the features of the data set.

Step4: Print the transformed features.
# Program:
Developed By: THARIKA S
Register No: 212222230159
# Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
# Basic Information:
```
df.head()
df.info()
df.info()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/04789ed7-32cd-485d-887d-c6a5f7ff6621)
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/23dfd6ba-221f-48ad-8ecd-473f4e74fef2)
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/dfa2136f-a4ba-4ba9-8b57-7f4e4aa22593)

# Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/ea5776da-afe1-4668-8aed-450a4bc7df04)
 ![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/662a748c-fa46-49e8-aeff-fb22c947ab7f)
 # Log Transformation:
 ```
 df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/8039da82-2295-446c-a164-b5f9769e70e8)
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/4a110153-8e80-44d8-82f0-f760eed18667)
# Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/3f3cb452-62a7-4208-bd96-5b3de364df22)
# SquareRoot Transformation: 
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/3f3cb452-62a7-4208-bd96-5b3de364df22)
# Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/e1822b75-5b73-4ddc-9d80-69da49d23ee2)
# Quantile Transformation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![image](https://github.com/tharikasankar/ODD2023-Datascience-Ex06/assets/119475507/baa4df96-896e-46ea-85dd-969ea37ad445)
# Result:
Thus feature transformation is done for the given dataset.



