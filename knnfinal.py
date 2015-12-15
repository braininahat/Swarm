import numpy as np
import cv2
from matplotlib import pyplot as plt

##input_letter = cv2.imread('/home/varun/opencv/Trails/Test/gridout.png')
##new_input_y = input_letter
##for ycount in range(1,25):
##    new_input_y = np.concatenate((input_letter, new_input_y), axis=0)
##new_input_x = new_input_y
##for xcount in range(1,25):
##    new_input_x = np.concatenate((new_input_y, new_input_x), axis=1)

##cv2.imwrite('/home/varun/opencv/Trails/gridout.png', new_input_x) #working till here 25x25 grid of C

img = cv2.imread('/home/varun/opencv/Trails/Test/gridout.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,10) for row in np.vsplit(gray,6)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x.reshape(-1,11250).astype(np.float32) # Size = (2500,400)
##test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(1,7)
train_labels = np.repeat(k,10)[:,np.newaxis]
##test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
##ret,result,neighbours,dist = knn.find_nearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
##matches = result==test_labels
##correct = np.count_nonzero(matches)
##accuracy = correct*100.0/result.size
##print accuracy

# save the data
np.savez('/home/varun/opencv/Trails/Test/knn_data.npz',train=train, train_labels=train_labels)

### Now load the data
##with np.load('/home/varun/opencv/Trails/knn_data.npz') as data:
##    print data.files
##    train = data['train']
##    train_labels = data['train_labels']
