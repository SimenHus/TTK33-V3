import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation

from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

import cv2
import pickle

import rosbag
from cv_bridge import CvBridge


DATA_FOLDER = '/mnt/c/Users/simen/Desktop/Prog/Python/TTK33/data/'
FILE = DATA_FOLDER + 'handheld/a_2024-10-15-09-19-31_3.bag'
TOTAL = '/mnt/c/Users/simen/Desktop/Prog/Python/TTK33/data/handheld/a_2024-10-15-09-19-31_3.bag'


ODOM_TOPIC = '/qualisys/morphy/odom'
VIDEO_TOPIC = '/tracking'



def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    X = np.reshape(X, [X.shape[0],-1]) # Needed to support N=1
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]



def drawCoordinateAxes(img, K, T, scale=1, labels=False):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = 0.8
    lineThickness = 3
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u, v = project(K, X)
    u, v = u.astype(int), v.astype(int)
    cv2.line(img, (u[0], v[0]), (u[1], v[1]), color=(255, 0, 0), thickness=lineThickness)
    cv2.line(img, (u[0], v[0]), (u[2], v[2]), color=(0, 255, 0), thickness=lineThickness)
    cv2.line(img, (u[0], v[0]), (u[3], v[3]), color=(0, 0, 255), thickness=lineThickness)
    if labels:
        cv2.putText(img, 'X', (u[1], v[1]), fontFace, fontScale, (255, 255, 255))
        cv2.putText(img, 'Y', (u[2], v[2]), fontFace, fontScale, (255, 255, 255))
        cv2.putText(img, 'Z', (u[3], v[3]), fontFace, fontScale, (255, 255, 255))

@dataclass
class Pose:
    """
    Object to contain and simplify Transformation matrix operations
    """
    R: 'np.ndarray[3, 3]' = field(default_factory=lambda: np.eye(3))
    t: 'np.ndarray[3]' = field(default_factory=lambda: np.zeros((3,)))

    def __post_init__(self):
        if self.R.shape[0] > 3:
            self.T = self.R
            self.R = self.T[:3, :3]
            self.t = self.T[:3, 3]
        else:
            self.T = np.zeros((4, 4))
            self.T[:3, :3] = self.R
            self.T[:3, 3] = self.t
            self.T[3, 3] = 1

    @property
    def inv(self) -> 'Pose':
        RT = self.R.T
        T = Pose(RT, -RT@self.t) # Faster processing than matrix inversion
        return T

    @property
    def pos(self) -> 'np.ndarray[3]':
        return self.t
    
    @property
    def rot(self) -> 'np.ndarray[3, 3]':
        return self.R
    
    @property
    def x(self) -> float:
        return self.t[0]
    
    @property
    def y(self) -> float:
        return self.t[1]
    
    @property
    def z(self) -> float:
        return self.t[2]

    def __repr__(self):
        return repr(self.T)

    def __matmul__(self, other):
        if type(other) == Pose:
            return Pose(self.T@other.T)
        else:
            return self.T@other
    

    @staticmethod
    def from_ros(msg) -> 'Pose':
        t = np.array([msg.position.x, msg.position.y, msg.position.z])
        R = Rotation.from_quat((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)).as_matrix()

        return Pose(R, t)
