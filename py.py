import pandas as pd
import numpy as np

data = pd.read_csv('./iris_data.txt', header=None)

# first suprise right here, indexing
data.head()
data[[0,1,2,3]].astype(int)
data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
data.columns

data['class'].unique()

mapper = {n:i for i,n in enumerate(data['class'].unique())}
mapper
data['class'] = data['class'].map(mapper)

data.tail()



data.iloc[1]
def knn_like(in_flower, data,k):
    neighbors = []

    for i,flower in enumerate(data.values):
        # print(type(flower[0]))
        dist = sum([(flower[j]-in_flower[j])**2 for j in range(len(in_flower))])**.5
        neighbors.append([i,dist])

    neighbors = sorted(neighbors, key= lambda x: (x[1]))[:k]
    print(neighbors)
    keys = [el[0] for el in neighbors]

    #print(data.iloc[keys])
    results = data.iloc[keys]
    return int(round(np.mean( results['class'] ),0))


#test_flower
data[data['class']==1].iloc[0]

new_iris = [7,3,5,1]

predicted_val = (knn_like(new_iris,data,6))
predicted_val
