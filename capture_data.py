from common import *


"""
Read a ROS bag and stores pixel center coordinates for the tracked target, as well as the pose for the drone.
"""
bag = rosbag.Bag(FILE) # Load rosbag
rosbag_result = bag.read_messages(topics=[VIDEO_TOPIC, ODOM_TOPIC]) # Read results for the desired topics

tracker = cv2.TrackerKCF_create() # Initialize tracker

bridge = CvBridge()
bbox = None # Bounding box for the tracked target
odom = None # Last odometry message
# Start timer
timer = cv2.getTickCount() # Used to display FPS
results = { # Results to be stored in data.pickle after capture
    'camera_pose': [],
    'detection_pixels': []
}
for topic, msg, t in rosbag_result:
    if topic == ODOM_TOPIC:
        odom = Pose.from_ros(msg.pose.pose) # Store last odometry message
        continue # Skip to next item in bag
    if odom is None: # Wait for first odometry message
        continue
    

    frame = bridge.imgmsg_to_cv2(msg, 'rbg8') # Convert ros image to cv2 image 

    if bbox is None: # Initialize bounding box for tracker
        bbox = cv2.selectROI(frame, False)
        tracker.init(frame, bbox)

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    current_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_time - timer)
    timer = current_time
    

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        centre = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        break # Break when first tracking failure happens

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Store tracked target center pixels and last pose of the drone
    results['camera_pose'].append(odom)
    results['detection_pixels'].append(centre)

    desired_FPS = 60
    desired_wait_ms = int(1000/desired_FPS)
    # Exit if ESC pressed
    k = cv2.waitKey(desired_wait_ms) & 0xff
    if k == 27 : break

# Write to file data.pickle
with open('data.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
bag.close()