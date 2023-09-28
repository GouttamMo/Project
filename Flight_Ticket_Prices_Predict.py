#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


data = pd.read_excel(r'C:\Users\dasgu\Desktop\Corzio\Data_Train.xlsx')


# In[9]:


data.head()


# In[10]:


data.info()


# In[11]:


data.shape


# In[12]:


data.count()


# In[14]:


data.dtypes


# In[15]:


data.describe()


# In[16]:


data.isna().sum()


# In[17]:


data[data['Route'].isna() | data['Total_Stops'].isna()]


# In[18]:


data.dropna(inplace = True)


# In[19]:


data.isna().sum()


# In[20]:


data.count()


# In[21]:


data.head()


# In[22]:


def convert_duration(duration):
    if len(duration.split())  == 2:
        hours = int(duration.split()[0][: -1])
        minutes = int(duration.split()[1][: -1])
        return hours * 60 + minutes
    else:
        return int(duration[: -1]) * 60


# In[23]:


data['Duration'] = data['Duration'].apply(convert_duration)
data.head()


# In[24]:


data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
data.dtypes


# In[25]:


data['Dep_Time_in_hours'] = data['Dep_Time'].dt.hour
data['Dep_Time_in_minutes'] = data['Dep_Time'].dt.minute
data['Arrival_Time_in_hours'] = data['Arrival_Time'].dt.hour
data['Arrival_Time_in_minutes'] = data['Arrival_Time'].dt.minute

data.head()


# In[26]:


data.drop(['Dep_Time', 'Arrival_Time'], axis = 1, inplace = True)
data.head()


# In[27]:


data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])
data.head()


# In[28]:


data['Date_of_Journey'].dt.year.unique()


# In[29]:


data['Day'] = data['Date_of_Journey'].dt.day
data['Month'] = data['Date_of_Journey'].dt.month

data.head()


# In[30]:


data.drop('Date_of_Journey', axis = 1, inplace = True)
data.head()


# In[31]:


data['Total_Stops'].value_counts()


# In[32]:


data['Total_Stops'] = data['Total_Stops'].map({
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
})


# In[33]:


data.head()


# In[34]:


data['Additional_Info'].value_counts()


# In[35]:


data.drop('Additional_Info', axis = 1, inplace = True)
data.head()


# In[36]:


data.select_dtypes(['object']).columns


# In[37]:


for i in ['Airline', 'Source', 'Destination', 'Total_Stops']:
    plt.figure(figsize = (15, 6))
    sns.countplot(data = data, x = i)
    ax = sns.countplot(x = i, data = data.sort_values('Price', ascending = True))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')
    plt.tight_layout()
    plt.show()
    print('\n\n')


# In[38]:


data['Airline'].value_counts()


# In[39]:


plt.figure(figsize = (15, 6))
ax = sns.barplot(x = 'Airline', y = 'Price', data = data.sort_values('Price', ascending = False))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')
plt.tight_layout()
plt.show()


# In[40]:


plt.figure(figsize = (15, 6))
ax = sns.boxplot(x = 'Airline', y = 'Price', data = data.sort_values('Price', ascending = False))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')
plt.tight_layout()
plt.show()


# In[41]:


data.groupby('Airline').describe()['Price'].sort_values('mean', ascending = False)


# In[42]:


Airline = pd.get_dummies(data['Airline'], drop_first = True)
Airline.head()


# In[43]:


data = pd.concat([data, Airline], axis = 1)
data.head()


# In[44]:


data.drop('Airline', axis = 1, inplace = True)
data.head()


# In[45]:


list1 = ['Source', 'Destination']
for l in list1:
    print(data[[l]].value_counts(), '\n')


# In[46]:


data = pd.get_dummies(data = data, columns = list1, drop_first = True)
data.head()


# In[47]:


route = data[['Route']]
route.head()


# In[48]:


data['Total_Stops'].value_counts()


# In[49]:


route['Route_1'] = route['Route'].str.split('→').str[0]
route['Route_2'] = route['Route'].str.split('→').str[1]
route['Route_3'] = route['Route'].str.split('→').str[2]
route['Route_4'] = route['Route'].str.split('→').str[3]
route['Route_5'] = route['Route'].str.split('→').str[4]

route.head()


# In[50]:


route.fillna('None', inplace = True)
route.head()


# In[51]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in range(1, 6):
    col = 'Route_' + str(i)
    route[col] = le.fit_transform(route[col])
    
route.head()


# In[52]:


route.drop('Route', axis = 1, inplace = True)
route.head()


# In[53]:


data = pd.concat([data, route], axis = 1)
data.head()


# In[54]:


data.drop('Route', axis = 1, inplace = True)
data.head()


# In[55]:


temp_col = data.columns.to_list()
print(temp_col, '\n')

new_col = temp_col[: 2] + temp_col[3:]
new_col.append(temp_col[2])
print(new_col, '\n')

data = data.reindex(columns = new_col)
data.head()


# In[56]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data = scaler.fit_transform(data)

data[0]


# In[57]:


from sklearn.model_selection import train_test_split as tts

x = data[:, : -1]
y = data[:, -1]


# In[58]:


x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.1, random_state = 69)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[59]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)


# In[60]:


from sklearn.metrics import mean_squared_error,r2_score

def metrics(y_true,y_pred):
    print(f'RMSE:',mean_squared_error(y_true,y_pred)** 0.5)
    print(f'R_Squared value:',r2_score(y_true,y_pred))
def accuracy(y_true,y_pred):
    errors =abs(y_true - y_pred)
    mape =100* np.mean(errors/y_true)
    accuracy = 100 - mape
    return accuracy


# In[62]:


y_pred = model.predict(x_test)


# In[63]:


metrics(y_test,y_pred)


# In[64]:


accuracy(y_test,y_pred)


# In[65]:


from sklearn.ensemble import RandomForestRegressor

model_random_forest = RandomForestRegressor(n_estimators= 500,min_samples_split=3)
model_random_forest.fit(x_train,y_train)


# In[71]:


pred_rf = model_random_forest.predict(x_test)


# In[72]:


metrics(y_test, pred_rf)


# In[73]:


accuracy(y_test, pred_rf)


# In[ ]:




