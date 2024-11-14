
from common import *

file = 'data.pickle' # Data file to load
with open(file, 'rb') as handle:
    b = pickle.load(handle) # Load data

K = np.loadtxt('./K.txt') # Camera intrinsic matrix
P = np.hstack( # Camera intrinsic matrix on the form 3x4 with last column being zeros
    [K, np.array([0, 0, 0]).reshape(3, 1)]
)


camera_trajectory = np.zeros([3, len(b['camera_pose'])]) # Store camera trajectory
object_triang = [] # Triangulated points array
poses: 'list[Pose]' = b['camera_pose'] # List of camera poses
pixels: 'list[tuple]' = b['detection_pixels'] # List of detection pixels

for i, pose in enumerate(poses): # Camera trajectory
    camera_trajectory[:, i] = pose.t


# Triangulation. All detections are triangulated against eachother and stored in object_triang
skip = 10 # Skip some samples that are close in time to eachother
for i, pixel in enumerate(pixels[:-1]):
    for j, other in enumerate(pixels[i+1:]):
        if not j % skip == 0: continue
        u1 = pixel # Camera 1 pixels
        u2 = other # Camera 2 pixels
        T1 = poses[i] # Camera 1 Pose
        T2 = poses[i+j] # Camera 2 Pose
        
        T_relative = (T1.inv@T2).T # Relative pose between cameras (.T is not transpose, it fetches the numpy matrix)
        P1 = P # Camera 1 P[K|0]
        P2 = P@T_relative # Camera 2 P[K|T]

        triang = cv2.triangulatePoints(P1, P2, u1, u2) # Triangulate
        triang = T1.inv@triang # Convert to world frame coordinates
        object_triang.append((triang[:3]/triang[3]).reshape((3,)))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def remove_outliers(data, thresh=1.0):
    # Calculate the mean and covariance matrix of the data, as well as the std deviation per dimension
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)

    # Invert the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Calculate the Mahalanobis distance for each data point
    distances = np.array([mahalanobis(x, mean, inv_cov_matrix) for x in data])

    # Identify points within the threshold distance (non-outliers)
    non_outliers = distances < thresh

    return data[non_outliers]


camera_X = camera_trajectory[0, :]
camera_Y = camera_trajectory[1, :]
camera_Z = camera_trajectory[2, :]
ax.plot(camera_X, camera_Y, camera_Z, label='Camera pos')

object_triang = np.array(object_triang)
object_triang = remove_outliers(object_triang)
object_X = object_triang.T[0, :]
object_Y = object_triang.T[1, :]
object_Z = object_triang.T[2, :]
ax.scatter3D(object_X, object_Y, object_Z, s=1, color='red', label='Measurements')

avg_X = np.mean(object_X)
avg_Y = np.mean(object_Y)
avg_Z = np.mean(object_Z)
ax.scatter3D(avg_X, avg_Y, avg_Z, label='Mean', color='green')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.legend()


# line, = ax.plot([], [], [])

# def update(i):
#     line.set_data(camera_X[:i], camera_Y[:i])
#     line.set_3d_properties(camera_Z[:i])
#     return line,

# ani = animation.FuncAnimation(fig, update, range(len(poses)), blit=False, interval=100, repeat=True)
plt.show()

# exit()


bag = rosbag.Bag(FILE)
rosbag_result = bag.read_messages(topics=[VIDEO_TOPIC, ODOM_TOPIC])

bridge = CvBridge()
timer = cv2.getTickCount()
last_odom = Pose()
frame_counter = 0

world_T = Pose()
mean_T = Pose(np.eye(3), np.array([avg_X, avg_Y, avg_Z]))


frame_width = 640
frame_height = 480
size = (frame_width, frame_height)
new_video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

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
    green = (0, 255, 0)
    X_camera = last_odom@np.array([avg_X, avg_Y, avg_Z, 1])
    world_origo = last_odom@np.array([0, 0, 1, 1])
    pixel_coords = project(K, X_camera)
    pixel_coords_w_o = project(K, world_origo)
    cv2.circle(frame, (int(pixel_coords[0]), int(pixel_coords[1])), 5, blue, 2)
    cv2.circle(frame, pixels[frame_counter], 5, red, 2)
    # drawCoordinateAxes(frame, K, (last_odom@world_T).T)
    # drawCoordinateAxes(frame, K, (last_odom@mean_T).T)

    # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.putText(frame, "Target", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)
    cv2.putText(frame, f"Estimate", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)

    # Display result
    cv2.imshow("Tracking", frame)
    new_video.write(frame)

    frame_counter += 1

    desired_FPS = 30
    desired_wait_ms = int(1000/desired_FPS)
    # Exit if ESC pressed
    k = cv2.waitKey(desired_wait_ms) & 0xff
    if k == 27 : break
new_video.release()
cv2.destroyAllWindows()