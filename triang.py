
from common import *
import cv2

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation

import rosbag
from cv_bridge import CvBridge

file = 'data.pickle'
with open(file, 'rb') as handle:
    b = pickle.load(handle)


# https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate

K = np.loadtxt('./K.txt')
P = np.hstack(
    [K, np.array([0, 0, 0]).reshape(3, 1)]
)


camera_trajectory = np.zeros([3, len(b['camera_pose'])])
object_triang = []
poses: 'list[Pose]' = b['camera_pose']
pixels: 'list[tuple]' = b['detection_pixels']

for i, pose in enumerate(poses):
    camera_trajectory[:, i] = pose.t

skip_frames = 1
for i, pixel in enumerate(pixels[:-1]):
    for j, other in enumerate(pixels[i+1:]):
        if not j % skip_frames == 0: continue
        u1 = pixel
        u2 = other
        T1 = poses[i]
        T2 = poses[i+j]
        
        T_relative = (T1@T2.inv).T
        P1 = P
        P2 = P@T_relative

        triang = cv2.triangulatePoints(P1, P2, u1, u2)
        triang = T1.inv@triang # Convert to world frame coordinates
        object_triang.append((triang[:3]/triang[3]).reshape((3,)))


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

def remove_outliers(data, thresh=1.0):           
    m = np.median(data)                            
    s = np.abs(data-m)                          
    return data[(s<np.median(s)*thresh).all(axis=1)]

# camera_X = camera_trajectory[0, :]
# camera_Y = camera_trajectory[1, :]
# camera_Z = camera_trajectory[2, :]
# # ax.plot(camera_X, camera_Y, camera_Z, label='Camera pos')

object_triang = np.array(object_triang)
object_triang = remove_outliers(object_triang).T
object_X = object_triang[0, :]
object_Y = object_triang[1, :]
object_Z = object_triang[2, :]
# ax.scatter3D(object_X, object_Y, object_Z, s=1, color='red', label='Measurements')

avg_X = np.mean(object_X)
avg_Y = np.mean(object_Y)
avg_Z = np.mean(object_Z)
# ax.scatter3D(avg_X, avg_Y, avg_Z, label='Mean', color='green')

# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
# ax.legend()


# line, = ax.plot([], [], [])

# def update(i):
#     line.set_data(camera_X[:i], camera_Y[:i])
#     line.set_3d_properties(camera_Z[:i])
#     return line,

# ani = animation.FuncAnimation(fig, update, range(len(poses)), blit=False, interval=100, repeat=True)
# plt.show()


bag = rosbag.Bag(FILE)
rosbag_result = bag.read_messages(topics=[VIDEO_TOPIC, ODOM_TOPIC])

bridge = CvBridge()
timer = cv2.getTickCount()
last_odom = Pose()
frame_counter = 0
for topic, msg, t in rosbag_result:    
    # color_encoding = 'bgr8'
    if topic == ODOM_TOPIC:
        last_odom = Pose.from_ros(msg.pose.pose) # Get odom
        continue
    if frame_counter >= len(pixels): break
    encodings = ['mono8', 'mono16', 'bgr8', 'bgra8', 'rgb8', 'rgba8']
    frame = bridge.imgmsg_to_cv2(msg, encodings[4])

    current_time = cv2.getTickCount()
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (current_time - timer)
    timer = current_time
    
    blue = (255, 0, 0)
    red = (0, 0, 255)
    X_camera = last_odom.inv@np.array([avg_X, avg_Y, avg_Z, 1])
    pixel_coords = project(K, X_camera)
    cv2.circle(frame, (int(pixel_coords[0]), int(pixel_coords[1])), 5, blue, 2)
    cv2.circle(frame, pixels[frame_counter], 5, red, 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    frame_counter += 1

    desired_FPS = 60
    desired_wait_ms = int(1000/desired_FPS)
    # Exit if ESC pressed
    k = cv2.waitKey(desired_wait_ms) & 0xff
    if k == 27 : break