import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import mixture
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def visualize_clusters(X_std,km,kmeans,x_label,y_label):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_std[km == 0,0], X_std[km == 0,1],
            c='green', label='cluster 1')
    plt.scatter(X_std[km == 1, 0], X_std[km == 1,1],
            c='blue', label='cluster 2')
    plt.scatter(X_std[km == 2,0], X_std[km == 2,1],
            c='yellow', label='cluster 3')
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', s=300,
            c='r', label='centroid')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Visualization of clustered data', fontweight='bold')
    ax.set_aspect('equal');
    plt.show()
    plt.close()

#Load Data Set
filename = 'breast-cancer-wisconsin.data.csv'
names = ['Id','ClumpThickness','CellSize','CellShape','MAdhesion','SingleECellSize','BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','class']
data = pandas.read_csv(filename,names=names)
to_drop = ['?']
df = data[~data['BareNuclei'].isin(to_drop)]
df = df.astype('int')



#Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train_encoded = le.transform(Y_train)
Y_test_encoded = le.transform(Y_validation)

#Dimensionality Reduction -  PCA
print "*****************Dimensionality Reduction Using PCA*****************"
pca = PCA(n_components =2)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_validation)

print X_train_std
#pca = PCA().fit(X_train_std)
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Pulsar Dataset Explained Variance')
#plt.show()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#K Cluster or K Means
print "*******************K Means Cluster*******************"
#Find optimal number of clusters using Elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_train_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCCS')
plt.show()
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
pred_y_train = kmeans.fit_predict(X_train_pca)
print pred_y_train
print "K Means Train Accuracy"
print accuracy_score(Y_train_encoded,pred_y_train)

visualize_clusters(X_train_pca,pred_y_train,kmeans,'petal-length','petal-width')



pred_y_test = kmeans.fit_predict(X_test_pca)
print pred_y_test
print "K Means Test Accuracy"
print accuracy_score(Y_test_encoded,pred_y_test)


#Expectation Maximization
giris = mixture.GaussianMixture(n_components=3,covariance_type = 'full')
giris.fit(X_train_pca)
labels_train  = giris.predict(X_train_pca)
print "Train Accuracy"
#print Y_train_encoded
#print labels_train

print accuracy_score(Y_train_encoded,labels_train)
visualize_clusters(X_train_pca,labels_train,giris,'petal-length','petal-width')


labels_test = giris.predict(X_test_pca)
print "Test Accuracy"
#print Y_test_encoded
#print labels_test
print accuracy_score(Y_test_encoded,labels_test)


#Run Neural model on dataset with dimesnionality reduction
nn_model = MLPClassifier(solver='lbfgs',activation='relu',alpha=1e-5,hidden_layer_sizes=(60,),learning_rate='constant',random_state=seed)
nn_model.fit(X_train_std,Y_train)
nn_model_predict = nn_model.predict(X_test_std)

print "$$$$$$$$$ Neural Network Accuracy $$$$$$$$$$$$$$"
print accuracy_score(Y_validation,nn_model_predict)

print "Neural Network Accuracy without Dimensionality Reduction"
nn_model.fit(X_train,Y_train)
xtest_predict = nn_model.predict(X_validation)

print accuracy_score(Y_validation,xtest_predict)
