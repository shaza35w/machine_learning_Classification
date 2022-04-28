#!/usr/bin/env python
# coding: utf-8

# # Classification using PCA, KNN, and logesitic regression
# 

# ### LFW dataset
# ###### The LFW dataset contains 13,233 images of faces collected from the web. This dataset consists of the 5749 identities with 1680 people with two or more images.

# In[1]:


from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt


# In[2]:


## load lfw dataset
lfw= fetch_lfw_people(min_faces_per_person=10)


# In[3]:


print(lfw.DESCR[:500])


# In[4]:


X=lfw.data
y=lfw.target
X.shape
## image size = 62* 47=2914


# ### Normalize the input features with mean zero and variance one

# In[5]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
n_X = sc_x.fit_transform(X)
n_X.shape
print(n_X)


# In[6]:


len(lfw.target_names)


# In[7]:


def plot_faces(images, n_row=2, n_col=5):
    w=47
    h=62
    """Helper function to plot a gallery of portraits"""
 
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.show()

#plot the average face and some samples from the dataset
plot_faces(X)


# ### Split data to 70% for training and 30% for testing. Use train_test_split from sklearn with random_state=5
# 

# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(n_X,y,test_size=.3,random_state=5)
from sklearn.metrics import accuracy_score


# ### implement a face recognition systems using scikit-learn on lfw dataset. These systems include:
# 1- Logestic regression model  (feed the images [raw features] into the logestic regression directly)<br> 
# 2- PCA + K_NN (number of principle components=100)<br>
# 3- PCA+ logestic regression( number of principle components=100)<br>
# 4- K-NN<br>

# In[9]:


# logistic
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(max_iter=5000)
lg.fit(x_train, y_train)
log_=lg.predict(x_test)
acc_log=accuracy_score(y_test, log_)
print(acc_log)


# In[10]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
neighbor=KNeighborsClassifier(n_neighbors=5)
neighbor.fit(x_train,y_train)
knn=neighbor.predict(x_test)
acc_knn=accuracy_score(knn, y_test)
print(acc_knn)


# In[11]:


#PCA_KNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), PCA(n_components=100), KNeighborsClassifier(n_neighbors=5))
model.fit(x_train, y_train)
p_knn=model.predict(x_test)
pca_knn=accuracy_score(p_knn, y_test)
print(pca_knn)


# In[12]:


#PCA_logistic
model = make_pipeline(StandardScaler(), PCA(n_components=100), LogisticRegression(max_iter=5000))
model.fit(x_train, y_train)
p_log=model.predict(x_test)
pca_log=accuracy_score(p_log, y_test)
print(pca_log)


# In[13]:


from tabulate import tabulate
table = [["logistic",acc_log*100],["KNN",acc_knn*100], ["PCA_KNN",pca_knn*100],["PCA_logistic",pca_log*100]]
print(tabulate(table,headers=["model","accuracy"],tablefmt="rst"))


# In[ ]:




