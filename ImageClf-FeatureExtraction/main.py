import pickle

from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

img2vec = Img2Vec()

data_dir = "/Users/berkantastan/Documents/GitHub/Computer-Vision-Projects/ImageClf-FeatureExtraction/data"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

data = {}
for j, dir_ in enumerate([train_dir, valid_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path = os.path.join(dir_, category, img_path)
            img = Image.open(img_path).convert('RGB')
            
            img_features = img2vec.get_vec(img) 
           
            features.append(img_features)
            labels.append(category)
            
    data[["training_data", "validation_data"][j]] = features
    data[["training_labels", "validation_labels"][j]] = labels

model = RandomForestClassifier()

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5)

grid_search.fit(data["training_data"], data["training_labels"])

print("En iyi parametreler:", grid_search.best_params_)

best_model = grid_search.best_estimator_

predictions = best_model.predict(data["validation_data"])
accuracy = accuracy_score(data["validation_labels"], predictions)

print(accuracy)

with open("./model.p","wb") as f:
    pickle.dump(best_model, f)
    f.close()








