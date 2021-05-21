#!/usr/bin/env python
# coding: utf-8

# # Problem 1.5 Decision Trees (elective)

# Uses the Decision Trees method and the learned lessons to mine the animal Zoo dataset available at http://archive.ics.uci.edu/ml/datasets/Zoo. The solution must be evaluated by the test dataset and real self-generated dataset.

# In[1]:


import pandas as pd 
df = pd.read_csv('/Users/nguyenphankhanhlinh/Downloads/zoo-4.csv')
df.head()


# In[2]:


df2 = pd.read_csv('/Users/nguyenphankhanhlinh/Downloads/class.csv')
df2.head()


# In[3]:


df3 = df.merge(df2,how='left',left_on='class_type',right_on='Class_Number')
df3.head(5)


# In[4]:


g = df3.groupby(by='Class_Type')['animal_name'].count()
g / g.sum() * 100


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df3['Class_Type'],label="Count",
             order = df3['Class_Type'].value_counts().index) #sort bars
plt.show()


# In[ ]:


# Using the FacetGrid from Seaborn, we can look at the columns to help us understand what features may be more helpful than others in classification


# In[6]:


feature_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',
                 'backbone','breathes','venomous','fins','legs','tail','domestic']

df3['ct'] = 1

for f in feature_names:
    g = sns.FacetGrid(df3, col="Class_Type",  row=f, hue="Class_Type")
    g.map(plt.hist, "ct")
    g.set(xticklabels=[])
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f)


# # The data using a heatmap

# In[7]:


gr = df3.groupby(by='Class_Type').mean()
columns = ['class_type','Class_Number','Number_Of_Animal_Species_In_Class','ct','legs'] #will handle legs separately since it's not binary
gr.drop(columns, inplace=True, axis=1)
plt.subplots(figsize=(10,4))
sns.heatmap(gr, cmap="YlGnBu")


# In[8]:


sns.stripplot(x=df3["Class_Type"],y=df3['legs'])


# # A decision tree works if we use all of the features available to us and training with 20% of the data

# In[9]:


#specify the inputs (x = predictors, y = class)
X = df[feature_names]
y = df['class_type'] #there are multiple classes in this column

#split the dataframe into train and test groups
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

#specify the model to train with
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) #ignores warning that tells us dividing by zero equals zero

#let's see how well it worked
pred = clf.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# In[10]:


df3[['Class_Type','class_type']].drop_duplicates().sort_values(by='class_type') #this is the order of the labels in the confusion matrix above


# In[11]:


imp = pd.DataFrame(clf.feature_importances_)
ft = pd.DataFrame(feature_names)
ft_imp = pd.concat([ft,imp],axis=1)
ft_imp.columns = ['Feature', 'Importance']
ft_imp.sort_values(by='Importance',ascending=False)


# # We reduced the training set size to 10%

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.1, test_size=.9) 

clf2 = DecisionTreeClassifier().fit(X_train, y_train)
pred = clf2.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# In[13]:


imp2 = pd.DataFrame(clf2.feature_importances_)
ft_imp2 = pd.concat([ft,imp2],axis=1)
ft_imp2.columns = ['Feature', 'Importance']
ft_imp2.sort_values(by='Importance',ascending=False)


# # Go back to 20% in the training group and focus on visible features of the animals.

# In[14]:


visible_feature_names = ['hair','feathers','toothed','fins','legs','tail']

X = df[visible_feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

clf3= DecisionTreeClassifier().fit(X_train, y_train)

pred = clf3.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf3.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# In[15]:


imp3= pd.DataFrame(clf3.feature_importances_)
ft = pd.DataFrame(visible_feature_names)
ft_imp3 = pd.concat([ft,imp3],axis=1)
ft_imp3.columns = ['Feature', 'Importance']
ft_imp3.sort_values(by='Importance',ascending=False)


# # If the dataset were larger, reducing the depth size of the tree would be useful to minimize memory required to perform the analysis. Below I've limited it to two still using the same train/test groups and visible features group as above.

# In[16]:


clf4= DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

pred = clf4.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf4.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# In[17]:


imp4= pd.DataFrame(clf4.feature_importances_)
ft_imp4 = pd.concat([ft,imp3],axis=1)
ft_imp4.columns = ['Feature', 'Importance']
ft_imp4.sort_values(by='Importance',ascending=False)


# In[18]:


columns = ['Model','Test %', 'Accuracy','Precision','Recall','F1','Train N']
df_ = pd.DataFrame(columns=columns)

df_.loc[len(df_)] = ["Model 1",20,.78,.80,.78,.77,81] #wrote the metrics down on paper and input into this dataframe
df_.loc[len(df_)] = ["Model 2",10,.68,.62,.68,.64,91]
df_.loc[len(df_)] = ["Model 3",20,.91,.93,.91,.91,81]
df_.loc[len(df_)] = ["Model 4",20,.57,.63,.57,.58,81]
ax=df_[['Accuracy','Precision','Recall','F1']].plot(kind='bar',cmap="YlGnBu", figsize=(10,6))
ax.set_xticklabels(df_.Model)


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[42]:


import pandas as pd
iris_data = pd.read_csv('/Users/nguyenphankhanhlinh/Downloads/zoo-4.csv')
iris_data.head()


# In[44]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit( iris.data, iris.target)


# In[45]:


import graphviz


# In[52]:


dot_data = tree.export_graphviz( clf, out_file= None, feature_names= iris.feature_names, class_names=iris.target_names, filled = True, rounded = True, special_characters= True)
graph = graphviz.Source(dot_data)
graph


# In[ ]:




