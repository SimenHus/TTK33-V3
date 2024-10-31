
from common import *

import pickle

with open('data.pickle', 'rb') as handle:
    b = pickle.load(handle)


for pose, coords in zip(b['detection_pixels'], b['camera_pose']):
    print(pose, coords)