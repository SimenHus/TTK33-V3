
from common import *

import pickle

with open('data.pickle', 'rb') as handle:
    b = pickle.load(handle)


# https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate

K = np.loadtxt('./K.txt')
Kinv = np.linalg.inv(K)

for coords, pose in zip(b['detection_pixels'], b['camera_pose']):
    pixels = np.array([*coords, 1])
    print(K@pixels)