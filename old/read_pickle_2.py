from common import *
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2

import rosbag
from cv_bridge import CvBridge

image_width = 640
image_height = 480
#camera_name: cam0
camera_matrix = np.array([[280.4816651778457, 0.0, 327.45578933301624], [0.0, 280.6856584239063, 222.83945115401625], [0.0, 0.0, 1.0]])
#distortion_model: equidistant
distortion_coefficients = np.array([-0.017952574136600025, 0.01808652312171992, -0.014472333463642921, 0.005505093849414231])
K_neg = np.linalg.inv(camera_matrix)



def triang(P_1, P_2, v_1, v_2):
    ##A*lambda=b

    A = np.array([[np.dot(v_1,v_1), -np.dot(v_1, v_2)],
                  [-np.dot(v_1, v_2), np.dot(v_2, v_2)]])
    
    b = np.array([np.dot(P_2-P_1, v_1),
                  -np.dot(P_2-P_1, v_2)])
    
    lam = np.linalg.solve(A,b)

    triang_point = ((P_1+lam[0]*v_1)+(P_2+lam[1]*v_2))/2
    return triang_point


with open('data.pickle', 'rb') as handle:
    b = pickle.load(handle)

x = []
y = []
z = []

x2 = []
y2 = []
z2 = []

drone_x = []
drone_y = []
P_data = []
points1_data = []

prev_point = []
prev_vector = []

skip_frames =200

index = 0
for pose, coords in zip(b['detection_pixels'], b['camera_pose']):
    #print(pose, coords)
    #cam_cord = np.array([[pose[0]],[pose[1]],[100]])

    pxl = np.array([pose[0], pose[1]])
    dst = cv2.undistortPoints(np.array([[pxl]], dtype=np.float32), camera_matrix, distortion_coefficients)
    #print(dst[0][0][0])
    cam_cord = np.array([[dst[0][0][0]],[dst[0][0][1]],[100]])
    homog_coords = np.append(dst[0][0], [1])
    ray1 = np.dot(coords.R, homog_coords - coords.t)
    origin1 = -np.dot(coords.R, coords.t)


    t = np.array([[coords.t[0]],[coords.t[1]],[coords.t[2]]])
    trefire = np.hstack((coords.R, t))
    P = camera_matrix@trefire
    P_data.append(P)

    points1 = np.array([[pose[0]], [pose[1]]], dtype=np.float32)
    points1_data.append(points1)

    
    XYZ = K_neg @ cam_cord
    ##drone
    R = coords.R
    pos = coords.t
    #pos = coords.t
    vect = R@XYZ.transpose()[0]
    

    fire = np.array([0,0,0])
    
    drone_x.append(pos[0])
    drone_y.append(pos[1])

    if index > skip_frames-1:
        try:
            fire = triang(prev_point[-skip_frames], pos, prev_vector[-skip_frames], vect)
            x.append(fire[0])
            y.append(fire[1])
            z.append(fire[2])
        except:
            pass
    if index > skip_frames-1:
        try:
            fire2 = cv2.triangulatePoints(P_data[-skip_frames], P, points1_data[-skip_frames], points1)
            points_3d = fire2 / fire2[3]
            #print(points_3d[1][0])
            x2.append(points_3d[0][0])
            y2.append(points_3d[1][0])
            z2.append(points_3d[2][0])
        except:
            pass

    
    
    
    prev_point.append(origin1)
    prev_vector.append(ray1)

    #prev_point.append(pos)
    #prev_vector.append(vect)
    index += 1
    

avg_X = np.mean(np.array(x))
avg_Y = np.mean(np.array(y))
avg_Z = np.mean(np.array(z))


bag = rosbag.Bag(FILE)
rosbag_result = bag.read_messages(topics=[VIDEO_TOPIC, ODOM_TOPIC])

poses: 'list[Pose]' = b['camera_pose']
pixels: 'list[tuple]' = b['detection_pixels']

K = np.loadtxt('./K.txt')

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
    X_camera = last_odom@np.array([avg_X, avg_Y, avg_Z, 1])
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