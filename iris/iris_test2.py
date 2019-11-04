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



def visualize_rawdata(X,Y,x_label,y_label):
    plt.figure(figsize=(6,6))
    plt.scatter(X,Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Visualization of raw data")
    plt.show()
    plt.close()


def visualize_clusters(X_std,km,kmeans,x_label,y_label):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_std[km == 0, 0], X_std[km == 0, 1],
            c='green', label='cluster 1')
    plt.scatter(X_std[km == 1, 0], X_std[km == 1, 1],
            c='blue', label='cluster 2')
    plt.scatter(X_std[km == 2, 0], X_std[km == 2, 1],
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

#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']

dataset = pandas.read_csv(url,names=names)
#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.fit_transform(X_validation)


#Visualize Raw Data
raw_names = ['sepal-length','sepal-width','petal-length','petal-width']
raw_data_train = pandas.DataFrame(X_train_std)
raw_data_train.columns = ['sepal-length','sepal-width','petal-length','petal-width']
visualize_rawdata(raw_data_train['sepal-length'],raw_data_train['sepal-width'],'sepal-length','sepal-width')

visualize_rawdata(raw_data_train['petal-length'],raw_data_train['petal-width'],'petal-length','petal-width')


le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train_encoded = le.transform(Y_train)
Y_test_encoded = le.transform(Y_validation)

#K Cluster or K Means
print "*******************K Means Cluster*******************"
#Find optimal number of clusters using Elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_train_std)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCCS')
plt.show()
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
pred_y_train = kmeans.fit_predict(X_train_std)
print pred_y_train
print Y_train_encoded
print "K Cluster Train Accuracy"
print accuracy_score(pred_y_train,Y_train_encoded)


#Plt scatter plot after K Means Clustering
#visualize_clusters(X_train_std,pred_y_train,kmeans,'sepal-length','sepal-width')
plt.scatter(X_train_std[:,2],X_train_std[:,3])
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s=300, c='red')
plt.xlabel('sepal-length')
plt.ylabel('sepal-width')
plt.legend()
plt.show()


pred_y_test = kmeans.fit_predict(X_validation)
print pred_y_test
print Y_test_encoded
print "K Cluster Test Accuracy"
print accuracy_score(pred_y_test,Y_test_encoded)


#Expectation Maximization
print "********************Expectation Maximization*******************"
giris = mixture.GaussianMixture(n_components=3,covariance_type = 'full')
giris.fit(X_train_std)
labels_train  = giris.predict(X_train_std)
print "Train Accuracy"

print accuracy_score(Y_train_encoded,labels_train)

labels_test = giris.predict(X_test_std)
print "Test Accuracy"
print Y_test_encoded
print labels_test
print accuracy_score(Y_test_encoded,labels_test)

#Dimensionality Reduction -  PCA
print "*****************Dimensionality Reduction Using PCA*****************"
pca = PCA(n_components =2)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_validation)

#fit and transform data
print "Train Before PCA Transform"
print X_train_std.shape
X_train_pca = pca.fit_transform(X_train_std) 
print "Train After PCA Transform"
print X_train_pca.shape
print "Test Before PCA Transform"
print X_test_std.shape
X_test_pca = pca.transform(X_test_std)
print X_test_pca.shape


#Dimensionality Reduction Using Random Projection
print "*****************Dimensionality Reduction Using Random Proection*****************"
randomprj = random_projection.SparseRandomProjection(n_components=2)
print "Before Random Projection"
print X_train.shape
X_train_new = randomprj.fit_transform(X_train_std)
print "After Random Projection"
print X_train_new.shape

#Feature Selection (Univariate feature Selection)
print "%%%%%%%%%%%%Univariate feature Selection%%%%%%%%%%%%%%%%%%"
kbest = SelectKBest(score_func=chi2,k=2)
Kbest_fit = kbest.fit(X_train,Y_train)
X_train_new = Kbest_fit.transform(X_train)
print X_train_new.shape

#Dimensionality Reduction using FastICA
print "*****************Dimensionality Reduction Using ICA*****************"
ica = FastICA(n_components=3,random_state=0)
print "Before ICA Transformation"
print X_train.shape
X_train_new = ica.fit_transform(X_train)
print "After ICA transformation"
print X_train_new.shape

