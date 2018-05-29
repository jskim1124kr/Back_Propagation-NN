import pandas as pd
import numpy as np
from keras.utils import np_utils



def load():
    load_data = pd.read_csv('Iris_cp.csv')

    data = load_data.values


    x_data = data[1:, 1:5].astype(float)
    y_data = data[1:, 5]

    y1 = []
    y2 = []
    y3 = []

    for i in y_data:
        if i == "Iris-setosa":
            y1.append(0)
        elif i == "Iris-versicolor":
            y2.append(1)
        elif i == "Iris-virginica":
            y3.append(2)



    y = np.concatenate([y1,y2,y3],0)

    y = np_utils.to_categorical(y)


    return x_data,y






