
# Classification with Iris data set

## Iris Data set

The Iris flower data set consists of 50 samples from each of three species of Iris:
- Iris setosa
- Iris virginica
- Iris versicolor

Four features were measured from each sample: 
- The length of the sepals
- The width of the sepals
- The length of the petals
- The width of the petals

<img src="iris_petal_sepal.png" style="width:250px"/>

## Load the dataset from scikit-learn datasets


```python
from sklearn.datasets import load_iris
```


```python
iris_data = load_iris()
```

### Data and labels


```python
iris_data.feature_names
```


```python
print(iris_data.target_names)
```


```python
X = iris_data.data
y = iris_data.target
```

#### Input is a m\*n matrix: 
- m: number of instances
- n: number of features


```python
import numpy as np
np.shape(X)
```


```python
X
```

#### Target is a vector of lenght m(number of instances)


```python
y
```

## Descriptive statistics

#### all data


```python
np.mean(X,0)
```


```python
np.std(X, 0)
```


```python
np.median(X, 0)
```


```python
np.min(X, 0)
```


```python
np.max(X, 0)
```

#### setosa


```python
np.mean(X[y==0], 0)
```


```python
np.std(X[y==0], 0)
```

#### versicolor


```python
np.mean(X[y==1], 0)
```


```python
np.std(X[y==1], 0)
```

#### virginica


```python
np.mean(X[y==2], 0)
```


```python
np.std(X[y==2], 0)
```

### Using the statistics to make a guess


```python
t1 = 2.5
t2 = 1.5
def guess(x):
    if(x[2] <= t1):
        return 0
    elif(x[3] <= t2):
        return 1
    return 2

def guess_all(X):
    return np.apply_along_axis(guess, 1, X)
```


```python
g = guess_all(X)
g == y
```


```python
sum(g == y)/len(y)
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(X[:,2], X[:,3], c=y)
plt.hold
plt.axvline(x=t1)
plt.axhline(y=t2)
```

## train/test split


```python
from sklearn.model_selection import train_test_split
```


```python
# to make the experiment replicable we use a constant random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 253)
```

## Five steps needed to work with sklearn models 

### 1. import the model


```python
from sklearn.neighbors import KNeighborsClassifier
```

### 2. create an instance 


```python
knn = KNeighborsClassifier(n_neighbors=1)
```

### 3. train the model using fit method


```python
knn.fit(X_train, y_train)
```

### 4. Make predictions using predict method


```python
p = knn.predict(X_test)
p
```

### 5. Evaluate


```python
y_test
```


```python
p == y_test
```


```python
sum(p == y_test)/len(y_test)
```

#### built-in evaluation metrics


```python
from sklearn import metrics
```


```python
metrics.accuracy_score(p, y_test)
```

## Using other models
[Here](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning) is a list of models that can be used in sklearn 


```python
# 1. import
from sklearn.naive_bayes import GaussianNB

# 2. instantiate
nb = GaussianNB()

# 3. train
nb.fit(X_train, y_train)

# 4. predict
p = nb.predict(X_test)

# 5. evaluate
metrics.accuracy_score(p, y_test)
```


```python
# 1. import
from sklearn.tree import DecisionTreeClassifier

# 2. instantiate
tree = DecisionTreeClassifier()

# 3. train
tree.fit(X_train, y_train)

# 4. predict
p = tree.predict(X_test)

# 5. evaluate
metrics.accuracy_score(p, y_test)
```


```python
# 1. import
from sklearn.svm import LinearSVC

# 2. instantiate
svm = LinearSVC()

# 3. train
svm.fit(X_train, y_train)

# 4. predict
p = svm.predict(X_test)

# 5. evaluate
metrics.accuracy_score(p, y_test)
```


```python
# 1. import
from sklearn.neural_network import MLPClassifier

# 2. instantiate
mlp = LinearSVC()

# 3. train
mlp.fit(X_train, y_train)

# 4. predict
p = mlp.predict(X_test)

# 5. evaluate
metrics.accuracy_score(p, y_test)
```


```python

```


```python

```
