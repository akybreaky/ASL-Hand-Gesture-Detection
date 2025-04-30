import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 

data_dictionary = pickle.load(open('./data.pickle', 'rb')) # Opens the file 'data.pickle' into read-binary mode

# Converts the lists into numpy arrays
data = np.asarray(data_dictionary['data']) 
labels = np.asarray(data_dictionary['labels'])

# Splitting the dataset
# 'x_train' and 'y_train' are used to train data and labels
# 'x_test' and 'y_test' are used for testing data and labels
# 'test_size' reserves 20% of the data for testing, 'shuffle' randomly shuffles the data before splitting, 'stratify' ensures that the class distrubution is balanced in both training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

# Training the classifier
model.fit (x_train, y_train)
y_predict = model.predict(x_test)

# Compares the predicted labels with the actual labels to calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

print('{}% of the samples were classified successfully.'.format(score * 100))

f = open('model.p', 'wb') # Opens a file named 'model.p' in wrtie-binary mode
pickle.dump({'model': model}, f)
f.close()