import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\opencv\sources\samples\data\digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# with np.load('knn_data.npz') as data:
#     print data.files
#     train = data['train']
#     train_labels = data['train_labels']

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

#Making numpy array
x = np.array(cells)

#  train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN. k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=5)

# Accuracy Test

matches = result==test_labels
correct = np.count_nonzero(matches)
np.savez('knn_data.npz',train=train, train_labels=train_labels)

accuracy = correct*100.0/result.size
print accuracy
