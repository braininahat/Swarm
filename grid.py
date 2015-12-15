import cv2
import numpy as np

new_input_x=cv2.imread('OutSample1 (1).png')
temp=new_input_x

def rowcat(letter):
    global new_input_x
    input_letter = cv2.imread('OutSample'+str(letter)+' (1).png')
    for xcount in range(1,10):#changed from 7 to 12
        i = 1+xcount
        new_input_x = cv2.imread('OutSample'+str(letter)+' ('+str(i)+').png')
        new_input_x = np.concatenate((input_letter,new_input_x), axis=1)
        input_letter= new_input_x
    return new_input_x

def colcat():
    column=np.concatenate((rowcat(1),rowcat(2),rowcat(3),rowcat(4),rowcat(5),rowcat(6)), axis=0)
    return column

cv2.imwrite('/home/varun/opencv/Trails/Test/gridout.png', colcat())
