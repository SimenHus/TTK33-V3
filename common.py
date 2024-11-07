DATA_FOLDER = '/mnt/c/Users/simen/Desktop/Prog/Python/TTK33/data/'
FILE = DATA_FOLDER + 'handheld/a_2024-10-15-09-18-03_2.bag'
TOTAL = '/mnt/c/Users/simen/Desktop/Prog/Python/TTK33/data/handheld/a_2024-10-15-09-19-31_3.bag'


ODOM_TOPIC = '/qualisys/morphy/odom'
VIDEO_TOPIC = '/tracking'


import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field


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

@dataclass
class Pose:
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
