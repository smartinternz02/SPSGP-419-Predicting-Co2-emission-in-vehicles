#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd #used for data manipulation
import numpy as np #used for numerical analysis
from collections import Counter as c # return counts
import seaborn as sns #used for data Visualization
import matplotlib.pyplot as plt
import missingno as msno #finding missing values
from sklearn.model_selection import train_test_split #splits data in random train and test array
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error#model performance
import pickle #Python object hierarchy is converted into a byte stream,
from sklearn.linear_model import LinearRegression #Regresssion ML algorithm


# ## Uploading Dataset

# In[2]:


data=pd.read_csv(r"C:\Users\LENOVO\internship\Dataset\Sample.csv") #loading the csv data


# In[3]:


data.head() #return you the first 5 rows values


# In[4]:


data.tail() #return you the last 5 rows values


# In[5]:


data.head(8) #return you the top 8 rows values


# In[6]:


data.drop('MODEL',axis=1,inplace=True)  # drop is used for dropping the column


# ### Renaming column names

# In[7]:


data.columns #return all the column names


# In[8]:


data.columns=['Make','MODEL.1', 'Vehicle_Class', 'Engine_Size', 'Cylinders',
       'Transmission', 'Fuel_Type', 'Fuel_Consumption_City',
       'Fuel_Consumption_Hwy', 'Fuel_Consumption_Comb(L/100 km)',
       'Fuel_Consumption_Comb(mpg)','CO2_Emissions'] # manually giving the name  of the columns
data.columns


# ### Info of the data

# In[9]:


data.info() #info will give you a summary of dataset


# In[10]:


data.describe()  # returns important values for continous column data


# ## Seeing Target, Nominal and Numerical Columns Count

# In[11]:


np.unique(data.dtypes,return_counts=True)


# ## Categoical Columns

# In[12]:


cat=data.dtypes[data.dtypes=='O'].index.values
cat


# ### Analysing the categorical columns

# In[13]:


for i in cat:
    print("Column :",i)
    print('count of classes : ',data[i].nunique())
    print(c(data[i]))
    print('*'*120)


# In[14]:


#here we are combininng the similar types of class into one class using where is for find
#and isin is for used for checking purpose
data["Transmission"] = np.where(data["Transmission"].
                            isin(["A4", "A5", "A3"]), "Automatic", data["Transmission"])
data["Transmission"] = np.where(data["Transmission"].isin(["M5", "M6"]), "Manual", data["Transmission"])
data["Transmission"] = np.where(data["Transmission"].
                        isin(["AS4", "AS5"]), "Automatic with Select Shift", data["Transmission"])
data["Transmission"] = np.where(data["Transmission"].
                isin(["AV"]), "Continuously Variable", data["Transmission"])
c(data['Transmission'])


# In[15]:


data["Fuel_Type"] = np.where(data["Fuel_Type"]=="Z", "Premium Gasoline", data["Fuel_Type"])
data["Fuel_Type"] = np.where(data["Fuel_Type"]=="X", "Regular Gasoline", data["Fuel_Type"])
data["Fuel_Type"] = np.where(data["Fuel_Type"]=="D", "Diesel", data["Fuel_Type"])
data["Fuel_Type"] = np.where(data["Fuel_Type"]=="E", "Ethanol(E85)", data["Fuel_Type"])
data["Fuel_Type"] = np.where(data["Fuel_Type"]=="N", "Natural Gas", data["Fuel_Type"])
c(data["Fuel_Type"])


# ## Numerical Columns

# In[16]:


data.dtypes[data.dtypes!='O'].index.values


# ## Checking Null values

# In[17]:


data.isnull().any()#it will return true if any columns is having null values


# In[18]:


data.isnull().sum() #used for finding the null values


# In[19]:


sns.heatmap(data.isnull(),cbar=False)


# In[20]:


msno.bar(data)
plt.show()


# ## Labeling the Categorical Columns

# In[21]:


data1=data.copy()
from sklearn.preprocessing import LabelEncoder #imorting the LabelEncoding from sklearn
x='*'
for i in cat:#looping through all the categorical columns
    print("LABEL ENCODING OF:",i)
    LE = LabelEncoder()#creating an object of LabelEncoder
    print(c(data[i])) #getting the classes values before transformation
    data[i] = LE.fit_transform(data[i]) # trannsforming our text classes to numerical values
    print(c(data[i])) #getting the classes values after transformation
    print(x*100)


# ## Data Visualization

# In[23]:


from tabulate import tabulate  # used for make data in a tabulated form
print(tabulate(pd.DataFrame(data1.Make.value_counts())))


# ## Feature:Make

# In[24]:


plt.figure(figsize=(19,5)); # give the figure size as width=19 and height = 5
#grouping Make with its count to find the top class 
data1.groupby("Make")["Make"].count().sort_values(ascending=False).plot(kind="bar")


# ## Feature:MODEL.1

# In[25]:


data1.groupby("MODEL.1")["MODEL.1"].count().sort_values(ascending=False)[:20].plot(kind="bar")


# ## Feature:Vehicle_Class

# In[26]:


data1.groupby('Vehicle_Class')['Vehicle_Class'].count().sort_values(ascending=False).plot(kind="bar")


# ## Feature:Transmission

# In[27]:


data1.groupby('Transmission')['Transmission'].count().sort_values(ascending=False).plot(kind='bar')


# ## Feature:Fule_Type

# In[28]:


data1.groupby('Fuel_Type')["Fuel_Type"].count().sort_values(ascending=False).plot(kind='bar')


# ## Make vc CO2_Emissions

# In[29]:


#grouping the Make and CO2_Emissions cloumns and storing top 20 classes.
MCO2=data1.groupby(['Make'])['CO2_Emissions'].mean().sort_values()[:20].reset_index()

plt.figure(figsize=(25,6))
#plotting the barplot between Make and CO2_Emissions column
sns.barplot(x = "Make",y="CO2_Emissions",data =MCO2 )


# ## Vehicle_CLass vs CO2_Emissions

# In[40]:


VC = data1.groupby('Vehicle_Class')['CO2_Emissions'].mean().sort_values(ascending=False)[:10].reset_index()
plt.figure(figsize=(20,6))
sns.barplot(x='Vehicle_Class',y='CO2_Emissions',data=VC)


# ## Transmission vs CO2_Emissions

# In[42]:


TC=data1.groupby('Transmission')['CO2_Emissions'].mean().sort_values(ascending=False)[:10].reset_index()
plt.figure(figsize=(20,6))
sns.barplot(x='Transmission',y='CO2_Emissions',data=TC)


# ## Fule_Type vs CO2_Emission

# In[43]:


FC=data1.groupby('Fuel_Type')['CO2_Emissions'].mean().sort_values()[:10].reset_index()
plt.figure(figsize=(20,6))
sns.barplot(x='Fuel_Type',y='CO2_Emissions',data=FC)


# ## Correlation

# In[44]:


corr = data.corr() #perform correlation between all continous features
plt.subplots(figsize=(16,16));
sns.heatmap(corr, annot=True, square=True) #plotting heatmap of correlations
plt.title("Correlation matrix of numerical features")
plt.tight_layout()
plt.show()


# ## Correlation of independent features with dependent variable

# In[45]:


plt.figure(figsize=(16,5))
corr["CO2_Emissions"].sort_values(ascending=True)[:-1].plot(kind="barh")


# ## Creating the Dependent and Independent variable

# In[46]:


x = data.drop(['CO2_Emissions','Fuel_Consumption_Comb(L/100 km)','MODEL.1'],axis=1) #independet features
x=pd.DataFrame(x)
y = data['CO2_Emissions'] #depenent feature
y=pd.DataFrame(y)


# ## Splitting dataset into train and test

# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape)
print(x_test.shape)


# ## Building our Model

# In[48]:


lr=LinearRegression() #creating object of LinearRegression model
lr=lr.fit(x_train,y_train) # fitting our model


# ## Predecting the results

# In[49]:


y_pred=lr.predict(x_test)
y_pred


# ## Checking the score of our model

# In[50]:


score = lr.score(x_test,y_test)
score


# In[51]:


from sklearn import metrics #importing the metrics library
print("MAE:",metrics.mean_absolute_error(y_test,y_pred)) #Mean Absolute Error
print("MSE:",metrics.mean_squared_error(y_test,y_pred)) # Mean Square Error
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred))) # Root Mean Square Error


# ## Dumping our model

# In[52]:


pickle.dump(lr,open("CO2.pkl",'wb'))


# In[ ]:




