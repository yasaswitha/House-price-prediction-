#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[54]:


import os
os.getcwd()


# In[55]:


df=pd.read_csv("Train.csv")
df.head()


# In[56]:


df.info()


# In[57]:


df.dtypes


# In[58]:


df.columns


# In[59]:


df.isnull().sum()


# In[60]:


df.describe()


# In[61]:


#splitting the address column into city and state column
df[['city','state']]=df["ADDRESS"].str.split(",",n=1,expand=True)
#dropping the address column
df=df.drop(columns=["ADDRESS"])


# In[62]:


#top 5 expensive houses
df.nlargest(n=5,columns='TARGET(PRICE_IN_LACS)')


# In[63]:


df.duplicated().any()


# In[64]:


#checking for correlation in data set
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')
plt.show()


# In[65]:


pd.plotting.scatter_matrix(df,color="red")
plt.figsize =(50,24)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# In[66]:


plt.scatter(df['SQUARE_FT'],df['TARGET(PRICE_IN_LACS)'])
plt.xlabel('Square footage')
plt.ylabel('Target price in lakhs')
plt.show()


# In[67]:


#create a box plot of price by BHK number
df.boxplot(column='TARGET(PRICE_IN_LACS)', by='BHK_NO.')
plt.title('price distribution by BHK number')
plt.xlabel('BHK number')
plt.ylabel('price in lakhs')
plt.show()


# In[68]:


plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=df['TARGET(PRICE_IN_LACS)'])
plt.title('property price by Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Price (in Lacs)')
plt.show()



# In[69]:


#Group the data by city and count the number of houses in each city
city_counts = df.groupby("state")["state"].count()
plt.bar(city_counts.index, city_counts.values)
plt.xlabel("State")
plt.ylabel("Number of Houses")
plt.title("Number of Houses in Each City")
plt.show()


# In[70]:


sns.boxplot(x='BHK_NO.', y='TARGET(PRICE_IN_LACS)', data=df)
# Set plot title and axis labels
plt.title('Property Prices by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price (in Lacs)')
# Show the plot
plt.show()


# #### property readiness to move
# sns.countplot(x='READY_TO_MOVE',data=df)
# plt.title('property readiness to move')
# plt.xlabel('readiness to move')
# plt.ylabel('Count')
# plt.show()

# In[71]:


grouped = df.groupby(['BHK_NO.', 'POSTED_BY'])
# Count the number of properties in each group
counts = grouped.size()
# Create a pie chart of property ownership by BHK number
for bhk in counts.index.levels[0]:
    bhk_counts = counts.loc[bhk]
    plt.figure()
    plt.pie(bhk_counts, labels=bhk_counts.index, autopct='%1.1f%%')
    plt.title(f'Property Ownership for {bhk} BHK')

# Show the plots
plt.show()


# In[92]:


import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score



# In[93]:


train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("Test.csv")


# In[94]:


X_train=train_df.drop('TARGET(PRICE_IN_LACS)',axis=1)
y_train=train_df ['TARGET(PRICE_IN_LACS)']


# In[95]:


X_train = train_df.drop('TARGET(PRICE_IN_LACS)', axis=1)
y_train = train_df['TARGET(PRICE_IN_LACS)']

# Preprocessing the train data
X_train = pd.get_dummies(X_train, columns=['POSTED_BY', 'BHK_OR_RK'])
X_train.drop(['ADDRESS'], axis=1, inplace=True)  # Drop ADDRESS column
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['SQUARE_FT', 'LONGITUDE', 'LATITUDE']] = scaler.fit_transform(X_train[['SQUARE_FT', 'LONGITUDE', 'LATITUDE']])

# Preprocessing the test data
X_test = pd.get_dummies(test_df, columns=['POSTED_BY', 'BHK_OR_RK'])
X_test.drop(['ADDRESS'], axis=1, inplace=True)  # Drop ADDRESS column
X_test[['SQUARE_FT', 'LONGITUDE', 'LATITUDE']] = scaler.transform(X_test[['SQUARE_FT', 'LONGITUDE', 'LATITUDE']])

# Splitting the training data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[96]:


svm = SVR(kernel='rbf')
svm.fit(X_train, y_train)


# In[97]:


y_train_pred = svm.predict(X_train)
y_val_pred = svm.predict(X_val)


# In[100]:


train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)


# In[101]:


print("Training set:")
print("Mean Squared Error:", train_mse)

print("Validation set:")
print("Mean Squared Error:", val_mse)


# Plotting the results
fig, ax = plt.subplots()
ax.scatter(y_train, y_train_pred, label="Training Set")
ax.scatter(y_val, y_val_pred, label="Validation Set")
ax.plot([0, max(y_train)], [0, max(y_train)], 'r--')
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("SVM Performance")
ax.legend()
plt.show()


# In[102]:


train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("Training set:")
print("R2 Score:", train_r2)
print("Validation set:")
print("R2 Score:", val_r2)


# In[ ]:




