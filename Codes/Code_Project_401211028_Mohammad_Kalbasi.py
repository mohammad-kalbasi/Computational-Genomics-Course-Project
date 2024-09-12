# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 17:00:43 2022

@author: mohammad
"""
#%% loading libraries pls make sure you have all this installed
import numpy as np
import pandas as pd
import random 
import pyreadr 
from mrmr import mrmr_classif
import random
from sklearn.svm import SVC
import seaborn as sns

from matplotlib import pyplot as plt







#%% 
def min_distance(x,y):
    _,temp_out = np.shape(y)
    out = np.zeros(temp_out,)
    for i in range(temp_out): 
        out[i] = np.sum(np.abs(y[:,i] - x))
    min_out = min(out)
    return min_out
#%% defining functions
def zscore_normalize(x):
    """
    z_score normalization for data input x
    inputs:
        x: input matrix, rows represent feature and column represnet sample
    output:
        normalized: normalized features
    """
    normalized = np.zeros(np.shape(x))
    _,n = np.shape(x)
    for i in range(n):
        normalized[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])
    return normalized

def file_reader(input_name = "lymphomadata"):
    """
    opening dataset and saving data as a matrix with it labels
    inputs:
        input_name: name of dataset
    output:
        X: feature matrix
        y: label matrix
        
    """
    if input_name == "leukaemia":
      df = pd.read_excel (input_name+".xlsx")
      print(df.head())
      header_names = list(df.columns.values)
      df_data = df[header_names[1:]]
      data = df_data.to_numpy()
      X,y = data[0:-2,:],data[-1,:]
      return X,y
    elif(input_name == "lymphomadata"):
        df = pd.read_excel (input_name+".xlsx")
        header_names = list(df.columns.values)
        df_data = df[header_names[1:]]
        data = df_data.to_numpy()
        X = data.T
        y = np.zeros(62)
        y[0:42] = 0
        y[42:] = 1
        return X,y
        
    elif(input_name == "lung"):
        result = pyreadr.read_r('lung.rda')
        df = result['lung']
        data = df.to_numpy()
        X,y = data[0:-2,:],data[-1,:]
        return X,y
    elif(input_name == "breast"):
        df = pd.read_csv(input_name+".csv")
        print(df.head())
        header_names = list(df.columns.values)
        df_data = df[header_names[1:]]
        X = df_data.to_numpy()
        y = np.zeros((60,))
        y[0:30] = 1
        return X,y
        
        
    else:
        return 1


   
      
def train_test_splitter(X,y,rate = 0.1,seed = 42):
    """
    train test split based on class population
    inputs:
        X: feature data
        y: labels
        rate: split rate for train and test
        seed: for reproducibility purpose(it may varry for different hardware)
    output:
        X_train:train data
        y_train:train label
        X_test:test data
        y_test:test label

    """
    yunique = np.unique(y)
    train_indics = []
    test_indics = []
    random.seed(seed)
    for i,label in enumerate(yunique):
        label_index = np.asarray(np.where(y == label)).squeeze()
        number_of_labels = len(label_index)
        test_number = int(np.round(rate*number_of_labels))
        random.shuffle(label_index)
        test_indics = np.hstack((test_indics,label_index[0:test_number])) # stacking index for different labrls
        train_indics = np.hstack((train_indics,label_index[test_number:-1]))
    X_train = X[:,train_indics.astype(int)]
    X_test = X[:,test_indics.astype(int)]
    y_train = y[train_indics.astype(int)]
    y_test = y[test_indics.astype(int)]
    return X_train,X_test,y_train,y_test
        
        
def center_calc(X,y):
    """
    calculating center of different classes
    input:
        X: feature data
        y:label data
    output:
        V:centers,from class with smallest number to biggest (1 ,2,...)
    """
    yunique = np.unique(y)
    m,_ = np.shape(X)
    V = np.zeros((m,len(yunique)))
    for i,label in enumerate(yunique):
         label_index = np.asarray(np.where(y == label)).squeeze()
         number_of_labels = len(label_index)
         V[:,i] = np.sum(X[:,label_index],axis = 1)/number_of_labels
    return V
        
def rbf_kernel(X,y,V,alpha,beta,C,method = 2):
    """
    calculating similarity masure based off single rbf kernel
    input:
        X: input feature
        V: centers
        y:input labels
        alpha: constant for rbf kernel
    output:
        dist: matrix of distance based on rbf kernel
    """
    dist = np.zeros(np.shape(X))
    _,m = np.shape(X)
    for i in range(m):
        x1 = X[:,i]
        x2 = V[:,int(y[i]-1)]
        if method == 1:
            dist[:,i] = 2-2*np.exp(-alpha*((x1-x2)**2))
        else:
            dist[:,i] = 2-(2*C*np.exp(-alpha*((x1-x2)**2)) + 2*(1-C)*np.exp(-beta*((x1-x2)**2)))
    return dist
def cost_calc(w,dist,delta):
    """
    calcularing cost function we want to minimize based on paper

    iputs:
    w : 
        weight vector.
    dist : 
        distance mat.
    delta : 
        coefficient for w in calculating cost.

 
    outputs
        j:calculated cost.

    """
    temp_mat = (w*(dist**2).T).T
    j = np.sum(temp_mat) + delta*(np.sum(w**2))
    return j
def delta_calculator(dist,w,a = 1):
    """
    calculating new delta
    inputs:
        dist:distance matrix
        w:weight vector.
    output:
        delta: weight for w in cost functions
        
    """
    delta = np.sum((w*(dist**2).T).T)/(np.sum(w**2))
    delta *= a
    return delta
def w_updater(dist,w,delta):
    """
    calculating new w's
    inputs:
        dist:distance matrix
        w:weight vector.
        delta: weight for w in cost functions
    output:
        w_new: new w vector     
    """
    l = len(w)
    w_new = np.zeros(l,)
    for k in range(l):
        chosen_w = dist[k,:]
        constant = np.sum(chosen_w**2)/l - chosen_w**2
        w_new[k] = (1/l) + (1/(2*delta))*np.sum(constant)
    return w_new
#%%


from sklearn.model_selection import train_test_split

# first choose which dataset we want to work on
#input_name = "lymphomadata"  best one
input_name = "breast"
X,y = file_reader(input_name)
_,m = np.shape(X)
for i in range(m):
    X[:,i] = X[:,i]*(i//5+1)
X_train_n,X_test_n,y_train,y_test = train_test_splitter(X,y,0.4,1)
X_train_n, X_test_n, y_train, y_test = train_test_split(X.T, y, test_size=0.4, random_state=42)
X_train = zscore_normalize(X_train_n.T)
X_test = zscore_normalize(X_test_n.T)
#%% 
V = center_calc(X_train,y_train)
m,_ = np.shape(V)
w = np.ones(m,)/m
dist = rbf_kernel(X_train,y,V,0.5,2.5,0.5,2)
# uncomment for second this method:
dist2 = rbf_kernel(X_train,1-y,V,1,2,0.5,2)
dist = dist/dist2
j = cost_calc(w,dist,1)
delta = delta_calculator(dist,w,1)
w = w_updater(dist,w,delta)

for counter in range(10009):
    delta = delta_calculator(dist,w,1)
    w = w_updater(dist,w,delta)
    j_new = cost_calc(w,dist,1)
    if (abs(j - j_new) < 0.0001):
        print(counter)
        break
 #       break;
    j = j_new
#%% plotting w
temp_dist = np.sum(dist,axis = 1)
temp_dist = temp_dist/np.max(temp_dist)
sorted_w = np.flip(np.sort(temp_dist))
plt.plot(sorted_w[0:200])
plt.xlabel('feature')
plt.ylabel('weight magnitude')

#%% we have different strategies to select features in this method
dist_sum = np.sum(dist,axis = 1)
dist_sorted = np.flip(np.argsort(dist_sum))
ordered_list = np.flip(np.argsort(w))
ther = 0.0001
chosen_index =np.where(w> ther)
index = np.where(w >=( np.max(w)/1.1 ))[0]
number_chosen = 30
chosen_index2 = dist_sorted[0:len(index)]
#X_train_chosen = X_train[index,:].squeeze()
#X_test_chosen= X_test[index,:].squeeze()

X_train_chosen = X_train[chosen_index2,:].squeeze()
X_test_chosen= X_test[chosen_index2,:].squeeze()

#%%
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
#number_chosen,_ = np.shape(X_train_chosen)

import timeit

start = timeit.default_timer()

#Your statements here

selector = SelectKBest(chi2, k=number_chosen)
X_kbest = selector.fit_transform(np.abs(X_train.T), y_train)
cols = selector.get_support(indices=True)
stop = timeit.default_timer()
print(stop-start)

#%%
X_train_chosen2 = X_train[cols,:]
X_test_chosen2= X_test[cols,:]


#%% predicting
from sklearn import svm
from sklearn.svm import LinearSVC

clf = LinearSVC(max_iter = 1000)
clf.fit(X_train_chosen.T, y_train)




#%% calculating train and test score
y_train_pred = clf.predict(X_train_chosen.T)
acc_train = np.sum(y_train_pred == y_train)/len(y_train)
print(f"accuracy on train data = {acc_train}.")


y_test_pred = clf.predict(X_test_chosen.T)
acc_test = np.sum(y_test_pred == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test}.")



#%% predicting
from sklearn.svm import LinearSVC
clf2 = LinearSVC(max_iter = 1000)
clf2.fit(X_train_chosen2.T, y_train)
y_train_pred2 = clf2.predict(X_train_chosen2.T)
acc_train2 = np.sum(y_train_pred2 == y_train)/len(y_train)
print(f"accuracy on train data with chi2 feature selection = {acc_train2}.")


y_test_pred2 = clf2.score(X_test_chosen2.T,y_test)
acc_test2 = np.sum(y_test_pred2 == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test2}.")


#%%
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_chosen.T, y_train)

y_train_pred = neigh.predict(X_train_chosen.T)
acc_train = np.sum(y_train_pred == y_train)/len(y_train)
print(f"accuracy on train data = {acc_train}.")


y_test_pred = neigh.predict(X_test_chosen.T)
acc_test = np.sum(y_test_pred == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test}.")


#%%
from matplotlib import pyplot as plt
w = clf.coef_
sorted_w = np.flip(np.argsort(w)[0])

plt.scatter(X_train_chosen2[0,:],X_train_chosen2[1,:],c = y_train)
plt.xlabel('PCA1')
plt.ylabel('PCA2')


#%% using MRMR

Xd = pd.DataFrame(X_train.T)
yd = pd.Series(y_train)

# select top 10 features using mRMR
selected_features = mrmr_classif(X=Xd, y=yd, K=10)

#%%
from sklearn.svm import SVC
X_train_n = X_train_n.T
X_test_n = X_test_n.T
X_train_chosen3 = X_train_n[selected_features,:]
X_test_chosen3= X_test_n[selected_features,:]
clf3 = SVC(random_state=42, kernel='linear')
clf3.fit(X_train_chosen3.T, y_train)
y_train_pred3 = clf3.predict(X_train_chosen3.T)
acc_train3 = np.sum(y_train_pred3 == y_train)/len(y_train)
print(f"accuracy on train data with mrmr feature selection = {acc_train3}.")


y_test_pred3 = clf3.predict(X_test_chosen3.T)
acc_test3 = np.sum(y_test_pred3 == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test3}.")



#%%
x_tick = ['normal','normal','normal','normal','normal','cancer','cancer','cancer','cancer','cancer']

yy = np.where(y_train == 1)[0]
X1 = X_train_chosen3[:,yy[0:5]]
yy = np.where(y_train == 2)[0]
X2 = X_train_chosen3[:,yy[0:5]]
X_tot = np.hstack((X1,X2))
sns.heatmap(X_tot,xticklabels = x_tick)




#%% testing feature reduction methods
def PCA_manual(X,c_num = 10):
  """
  function for calculating PCA transform
  inputs:
    X:input matrix, columns are our samples
    c_num:number of components
  output:
    encoded: encoded data
    recons: reconstructed data
    eigenVectors: chosen eigen vectors
    X_m: mean of data for each feature

  """
  X_m = np.mean(X,axis = 1).reshape(-1,1)
  X = X - X_m
  C = np.matmul(X,X.T)
  eigenValues , eigenVectors = np.linalg.eigh(C)
  idx = eigenValues.argsort()[::-1]   
  eigenValues = eigenValues[idx]
  eigenVectors = eigenVectors[:,idx]
  eigenVectors = eigenVectors[:,0:c_num]
  encoded = np.matmul(eigenVectors.T,X)
  recons = np.matmul(eigenVectors,encoded) + X_m
  return encoded,recons,eigenVectors,X_m
#%% plotting features

encoded,recons,eigenVectors,X_m = PCA_manual(X_train_chosen3,2)
plt.scatter(encoded[0,:],encoded[1,:],c = y_train)

X_test_3 = X_test_chosen3 - X_m
projected = np.matmul(eigenVectors.T,X_test_3)
plt.scatter(projected[0,:],projected[1,:],c = y_test)

plt.title('first two principal components')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


#%%
from sklearn.svm import SVC
clf3 = SVC(random_state=42, kernel='rbf')
clf3.fit(encoded.T, y_train)
y_train_pred3 = clf3.predict(encoded.T)
acc_train3 = np.sum(y_train_pred3 == y_train)/len(y_train)
print(f"accuracy on train data with anova feature selection = {acc_train3}.")


y_test_pred3 = clf3.predict(projected.T)
acc_test3 = np.sum(y_test_pred3 == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test3}.")



#%% testing feature reduction with pca
encoded,recons,eigenVectors,X_m = PCA_manual(X_train,2)
plt.scatter(encoded[0,:],encoded[1,:],c = y_train)

X_test_2 = X_test - X_m
projected = np.matmul(eigenVectors.T,X_test_2)
plt.scatter(projected[0,:],projected[1,:],c = y_test)


plt.show()
#%%

from sklearn.svm import SVC
clf3 = SVC(random_state=42, kernel='linear')
clf3.fit(encoded.T, y_train)
y_train_pred3 = clf3.predict(encoded.T)
acc_train3 = np.sum(y_train_pred3 == y_train)/len(y_train)
print(f"accuracy on train data with anova feature selection = {acc_train3}.")


y_test_pred3 = clf3.predict(projected.T)
acc_test3 = np.sum(y_test_pred3 == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test3}.")


#%%
def MSE(A,B):
  
  """
  calculating mse error between two matrix
  """
  error = np.square(np.subtract(A, B)).mean()
  return error




#%%
def NMF(A,j,scale,max_iter,thr):
  """
  function for calculating non negative matrix factorization
  we want to estimat A = BC + error, and all of B and C values are positive
  inputs:
    A:input matrix
    j:number of elements for feature reduction
    scale:scaling random initialization
    thr: threshold for minimum change
  output:
    B and C matrix
  """
  epsilon = 1e-6
  n1,n2 = np.shape(A)
  B = scale*np.random.rand(n1,j)
  C = scale*np.random.rand(j,n2)
  for i in range(max_iter):
    num = np.matmul(B.T,A)
    den = np.matmul(B.T,np.matmul(B,C))
    update_C = num/(den + epsilon)
    C_old = C
    C = C*update_C
    num = np.matmul(A,C.T)
    den = np.matmul(B,np.matmul(C,C.T))
    update_B = num/(den+epsilon)
    B_old = B
    B = B*update_B
    mse_error = MSE(np.matmul(B,C),np.matmul(B_old,C_old))
    if mse_error < thr:
      break
  return B,C
    

#%%
B,C = NMF(X_train_n,2,10,10000,0.0)
B_test = np.matmul(X_test_n,np.linalg.pinv(C))
#%%
plt.scatter(B[:,0],B[:,1],c = y_train)
plt.scatter(B_test[:,0],B_test[:,1],c = y_test)

#%% Genetic algorithm
def gene_correction(input_vec):
    """
    function for making sure at least on gene is selected
    """
    len_vec = len(input_vec)
    if ( np.sum(input_vec ==0) == len_vec):
        input_vec[ 0 : random.randint(2, len_vec-1)] = 1
        np.random.shuffle(input_vec)
    return input_vec
        

  
    
def mutation(input_vec,number_chosen):
    """
    function for apllying mutation on input gene
    
    inputs:
        input_vec: input gene
        number_chosen: number of gene we mutate
    outputs:
        output: mutated gene
    
    """
    len_vec = len(input_vec)
    if number_chosen > len_vec:
        number_chosen = len_vec
    random_number = np.arange(len_vec)
    np.random.shuffle(random_number)
    output = np.zeros((len(input_vec)))
    output[random_number[0:number_chosen]] = 1 - input_vec[random_number[0:number_chosen]]
    output[random_number[number_chosen:]] = input_vec[random_number[number_chosen:]]
    output = gene_correction(output)
    return output
    
def cross_over(input_vec1,input_vec2):
    len_vec = len(input_vec1)
    random_chosen =  random.randint(1, len_vec-1)
    output1 = np.zeros((len_vec))
    output2 = np.zeros((len_vec))
    output1[0:random_chosen] = input_vec1[0:random_chosen]
    output1[random_chosen:] = input_vec2[random_chosen:] 
    output2[0:random_chosen] = input_vec2[0:random_chosen]
    output2[random_chosen:] = input_vec1[random_chosen:] 
    output1 = gene_correction(output1)
    output2 = gene_correction(output2)
    return output1 , output2


def genetic_alg(X_train,y_train,X_test,y_test,k1,k2,k3,k4,thr,max_iter,population,kernel_chosen):
    """
    function for finding best features based on genetic algorithm
    inputs:
        X_train,y_train,X_test,y_test: inputs of data for train and test
        k1,k2,k3,k4: probability controling variables
        thr: threshould for desired accuracy
        max_iter : max iteration befor breaking loop
        population: number of samples we keep
        kernel_chosen : choesn kernel for svm
    return:
        


    """
    n2,n1 = np.shape(X_train)
    parent = np.zeros((population,n1))
    score_parent = np.zeros((population,))
    for i in range(population):
        parent[i,:] = mutation( parent[i,:],random.randint(2, n1-1))
        clf = SVC(random_state=42, kernel=kernel_chosen)
        clf.fit(X_train[:,parent[i,:] == 1], y_train)


        score_parent[i] = clf.score(X_test[:,parent[i,:] == 1],y_test)
    
    random_chosen = mutation( parent[random.randint(0, population-1),:],random.randint(2, n1-1))
    student = random_chosen 
    k_mutate = k2
    k_corss = k4
    random_create = np.random.uniform(0,1,population)
    for i in range(population):
        if random_create[i] < k_mutate:
            temp_create = mutation( parent[i,:],random.randint(2, n1-1))
            student = np.vstack((student,temp_create))
        if random_create[i] < k_corss:
            temp_create,temp_create2 = cross_over( parent[i,:],parent[random.randint(0, population-1),:])
            student = np.vstack((student,temp_create))
    
    n1,n2 = np.shape(student)
    score_student = np.zeros((n1,))
    for i in range(n1):
        clf = SVC(random_state=42, kernel=kernel_chosen)
        clf.fit(X_train[:,student[i,:] == 1], y_train)
        score_student[i] = clf.score(X_test[:,student[i,:] == 1],y_test)
    f_prime = np.max(score_student)
    score_all = np.hstack((score_parent,score_student))
    feature_all = np.vstack((parent,student))
    f_max = np.max(score_all)
    f_avg = np.mean(score_all)
    if f_prime <f_avg:
        k_mutate = k2
        k_corss = k4
    else:
        k_mutate = k1*(f_prime - f_avg)/(f_max - f_avg)
        k_mutate = k1*(f_prime - f_avg)/(f_max - f_avg)
    score_max = np.flip(np.argsort(score_all))
    parent = feature_all[score_max[0:population],:]
    score_parent = score_all[score_max[0:population]]
        
    for counter in range(max_iter):
        random_chosen = mutation( parent[random.randint(0, population-1),:],random.randint(2, n1-1))
        student = random_chosen 
        random_create = np.random.uniform(0,1,population)
        for i in range(population):
            if random_create[i] < k_mutate:
                temp_create = mutation( parent[i,:],random.randint(2, n1-1))
                student = np.vstack((student,temp_create))
            if random_create[i] < k_corss:
                temp_create,temp_create2 = cross_over( parent[i,:],parent[random.randint(0, population-1),:])
                student = np.vstack((student,temp_create))
        
        n1,n2 = np.shape(student)
        score_student = np.zeros((n1,))
        for i in range(n1):
            clf = SVC(random_state=42, kernel=kernel_chosen)
            clf.fit(X_train[:,student[i,:] == 1], y_train)
            score_student[i] = clf.score(X_test[:,student[i,:] == 1],y_test)
        f_prime = np.max(score_student)
        score_all = np.hstack((score_parent,score_student))
        feature_all = np.vstack((parent,student))
        f_max = np.max(score_all)
        f_avg = np.mean(score_all)
        if f_prime <f_avg:
            k_mutate = k2
            k_corss = k4
        else:
            k_mutate = k1*(f_prime - f_avg)/(f_max - f_avg)
            k_mutate = k1*(f_prime - f_avg)/(f_max - f_avg)
        score_max = np.flip(np.argsort(score_all))
        parent = feature_all[score_max[0:population],:]
        score_parent = score_all[score_max[0:population]]
        if(np.max(score_all)>thr):
            break
         
        
        
    return parent,score_parent

    
    
    
#%% now testing function, for this purpose we mix features with worst features and see if model learns
dist_sum = np.sum(dist,axis = 1)
dist_sorted = np.flip(np.argsort(dist_sum))
ordered_list = np.flip(np.argsort(w))
ther = 0.0001
chosen_index =np.where(w> ther)
index = np.where(w >=( np.max(w)/1.1 ))[0]
number_chosen = 30
chosen_index_good = dist_sorted[0:number_chosen]
dist_sorted = np.flip(dist_sorted)
chosen_index_bad = dist_sorted[0:number_chosen]
chosen_index = np.hstack((chosen_index_good,chosen_index_bad))
X_train_chosen = X_train[chosen_index,:].squeeze()
X_test_chosen= X_test[chosen_index,:].squeeze()
parent,score_parent = genetic_alg(X_train_chosen.T,y_train,X_test_chosen.T,y_test,0.9,0.6,0.1,0.01,12,1000,30,'linear')
     


#%% autoencoder
import keras
from keras import layers
import tensorflow as tf
class auto_encoder:
  def __init__(self, bottleneck_dim,num_epoch,dim_input):
    self.bottleneck_dim = bottleneck_dim
    self.num_epoch = num_epoch
    self.dim_input = dim_input
  def creat_model(self):
    input = layers.Input(shape=(self.dim_input,))
    e = layers.Dense(1280,activation='selu',name = 'encoder1')(input)
    #e = layers.BatchNormalization()(e)
    # encoder level 2
    e = layers.Dense(640,activation='selu',name = 'encoder2')(e)
    #e = layers.BatchNormalization()(e)
    # bottleneck
    bottleneck = layers.Dense(self.bottleneck_dim,activation='selu',name = 'bottleneck')(e)

    d = layers.Dense(640,activation='selu',name = 'decoder1')(bottleneck)
    #d = layers.BatchNormalization()(d)
    # decoder level 2
    d = layers.Dense(1280,activation='selu',name='decoder2')(d)
    #d = layers.BatchNormalization()(d)
    # output layer
    output = layers.Dense(self.dim_input, activation='relu')(d) #outputs are always positive, because of that we use relu for output activation
    self.model = keras.Model(input, output)
  def train_model(self,X_train,y_train,X_test,y_test):
    self.model.compile(optimizer='adam', loss='mse',)

    history = self.model.fit(X_train, y_train,
                    epochs=self.num_epoch,
                    batch_size=256,
                    shuffle=True,
                  
                    validation_data=(X_test, y_test))
    return history,self.model
  
    

#%%
n1,n2 = np.shape(X_train)
encoder_model = auto_encoder(25,200,n1)
encoder_model.creat_model()
history,autoencoder = encoder_model.train_model(X_train.T,X_train.T,X_test.T,X_test.T)

#%%
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output) # if we want encoder part separatly
train_encode = encoder(X_train.T)
test_encode = encoder(X_train.T)

from sklearn.svm import SVC
clf6 = SVC(random_state=42, kernel='linear')
clf6.fit(train_encode, y_train)
y_train_pred6 = clf6.predict(train_encode)
acc_train3 = np.sum(y_train_pred6 == y_train)/len(y_train)
print(f"accuracy on train data with AE feature reduction = {acc_train3}.")


y_test_pred6 = clf3.predict(test_encode)
acc_test3 = np.sum(y_test_pred6 == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test3}.")


#%% svm backward
X_train_t = X_train.T
X_test_t = X_test.T
n1,n2 = np.shape(X_train_t)
ar_chosen = np.arange(n2)
n_len = len(ar_chosen)
while(n_len<50):
  clf = SVC(random_state=42, kernel='linear')
  x_temp = X_train_t[:,ar_chosen]
  clf.fit(x_temp, y_train)
  coef = clf.coef_
  ar_chosen = np.delete(ar_chosen,np.argmin(np.abs(coef)))
  n_len = len(ar_chosen)


x_chosen = X_train_t[:,ar_chosen]
x_t_chosen = X_test_t[:,ar_chosen]

from sklearn.svm import SVC
clf6 = SVC(random_state=42, kernel='linear')
clf6.fit(x_chosen, y_train)
y_train_pred6 = clf6.predict(x_chosen)
acc_train3 = np.sum(y_train_pred6 == y_train)/len(y_train)
print(f"accuracy on train data with AE feature reduction = {acc_train3}.")


y_test_pred6 = clf3.predict(X_test_t)
acc_test3 = np.sum(y_test_pred6 == y_test)/len(y_test)
print(f"accuracy on test data = {acc_test3}.")
