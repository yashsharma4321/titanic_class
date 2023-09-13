#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Investigation

# # Introduction
# 
This Jupyter Notebook investigates the Titanic dataset provided by Kaggle.

The objective of this investigation is to determine chances of survival of a passenger based on:

.Age
.Gender
.Number of siblings and spouses aboard
.Number of parents and children aboard
This investigation therefore answers the following question: "How likely a passenger would survive, based on age, gender, ticket class and number of siblings and spouses aboard and number of parents and children aboard"

To begin with, the dataset is loaded into a Pandas Dataframe, and its first few records are viewed.
# In[1]:


import csv
import pandas as pd
titanic_df = pd.read_csv('titanic.csv', quoting=csv.QUOTE_MINIMAL, skiprows=[0],
                         names=['passenger_id', 'survived', 'class', 'name', 'sex', 'age',
                                'sib_sp', 'par_ch', 'ticket_id', 'fare', 'cabin', 'port'])
titanic_df.head()


# # Data Wrangling
To begin with, attributes that are not considered in the investigation (passenger_id, name, titcket_id, fare, cabin and port) can be removed from the dataset.
# In[2]:


titanic_df = titanic_df.drop(['passenger_id', 'name', 'ticket_id', 'fare', 'cabin', 'port'], axis=1)
titanic_df.head()

Next, to ensure that the dataset is ready for analysis, check whether any attributes have missing values.
# In[3]:


titanic_df['survived'].isnull().sum()


# In[4]:


titanic_df['age'].isnull().sum()


# In[6]:


titanic_df['sex'].isnull().sum()


# In[5]:


titanic_df['sib_sp'].isnull().sum()


# In[7]:


titanic_df['par_ch'].isnull().sum()

The age attribute seems to have missing values; 177 to be exact. These missing values could be ignored during the analysis.
# In[8]:


titanic_df = titanic_df[titanic_df['age'].notnull()]
titanic_df['age'].isnull().sum()


# # Survivors
The dataset provides details of passengers aboard Titanic, and wether or not they survived. Objective of this investigation is to determine chances of survival of a passenger based on their age, sex, class, number of siblings and parents aboard. Survivors are identified using survived attribute with values 0 and 1, representing non survivors and survivors respectively.
# # Survival based on Age

# In[9]:


survivors = titanic_df.groupby('survived')['age']
survivors.describe()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

ax1, ax2 = survivors.plot(kind='hist', legend=True, alpha=0.6, bins=range(0, 90, 10))
ax1.set_xlabel('Age (Years)')
ax1.legend(['No', 'Yes'], title='Survived', loc='upper right')
ax2.set_ylabel('Frequency')
plt.title('Comparison of survivors v/s non survivors based on age')

From the results it can be concluded that age is not a determining factor for survival as the shape of the histogram is almost the same for both survivors and non survivors with the exception of children.
# # Survival based on Gender

# In[11]:


survivors = titanic_df.groupby('survived')['sex']
survivors.describe()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

survivors = pd.crosstab(titanic_df['survived'], titanic_df['sex'])
survivors.plot(kind='bar', legend=True)
plt.xlabel('Survived?')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.ylabel('Frequency')
plt.legend(['Female', 'Male'], title='Sex', loc='upper right')
plt.title('Comparison of survivors v/s non survivors based on sex')


# # Survival based on Ticket Class

# In[13]:


survivors = titanic_df.groupby('survived')['class']
survivors.describe()


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

survivors = pd.crosstab(titanic_df['survived'], titanic_df['class'])
survivors.plot(kind='bar')
plt.xlabel('Survived?')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.ylabel('Frequency')
plt.legend(['1st', '2nd', '3rd'], title='Ticket Class', loc='upper right')
plt.title('Comparison of survivors v/s non survivors based on ticket class')

From the results, it can be concluded that first and second class passengers had higher chances of survival than third class passengers.
# # Survival based on Siblings and Spouses aboard

# In[15]:


survivors = titanic_df.groupby('survived')['sib_sp']
survivors.describe()


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

ax1, ax2 = survivors.plot(kind='hist', bins=range(0, 8, 1), legend=True, alpha=0.6)
ax1.set_xlabel('No. of Siblings and Spouses Aboard')
ax1.legend(['No', 'Yes'], title='Survived', loc='upper right')
plt.title('Comparison of survivors v/s non survivors based on siblings and spouses aboard')

From the results it can be concluded that having siblings and spouses aboard is not a factor determining survival of a passenger.
# # Survival based on Parents and Children aboard

# In[17]:


survivors = titanic_df.groupby('survived')['par_ch']
survivors.describe()


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

ax1, ax2 = survivors.plot(kind='hist', legend=True, bins=range(0, 7, 1), alpha=0.6)
ax1.set_xlabel('No. of Parents and Children Aboard')
ax1.legend(['No', 'Yes'], title='Survived?', loc='upper right')
plt.title('Comparison of survivors v/s non survivors based on parents and children aboard')

From the results, it can be concluded that having parents and children aboard is not a factor determining survival of a passenger.
# # Limitations
The main limitation of this dataset investigation is its missing values.
# # Conclusion
From the analysis, it can be concluded that factors determining a passengers survival are:

.Ticket class of the passenger
.Gender of the passenger

Factors that doesn't determine a passengers survival are:

.Siblings and spouses aboard with the passenger
.Parents and children aboard with the passenger
.Age of the passenger