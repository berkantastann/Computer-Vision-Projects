import os
from skimage.io import imread
from skimage.transform import resize
from numpy import asarray
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

input_dir = "/ImageClassification/clf-data"
categories = ["empty", "not_empty"]

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (30,30))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

classifier = SVC()

parameters = {"gamma":[0.01,0.001,0.0001],"C":[1,10,100,1000],"kernel":["linear","rbf","sigmoid"]}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters)

grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_
y_predict = best_estimator.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

pickle.dump(best_estimator, open("/ImageClassification/model.p", "wb"))
