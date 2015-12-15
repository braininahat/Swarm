import cv2
import numpy as np

# Now load the data
with np.load('/home/varun/opencv/Trails/Test/knn_data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']

imgread = cv2.imread('OutSample1 (1).png')
img = cv2.resize(imgread,(150,75))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

test = gray.reshape(-1,11250).astype(np.float32)

for i in range(1,7):
    test_label = np.array([i])

    knn = cv2.KNearest()
    knn.train(train,train_labels)
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    matches = result==test_label
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print i,accuracy
