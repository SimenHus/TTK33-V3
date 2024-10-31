import cv2
import rosbag
from cv_bridge import CvBridge

from common import *

import pickle


def main() -> None:
    bag = rosbag.Bag(FILE)
    rosbag_result = bag.read_messages(topics=[VIDEO_TOPIC, ODOM_TOPIC])
 
    # Instead of MIL, you can also use
    # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker = cv2.TrackerKCF_create()

    bridge = CvBridge()
    bbox = None
    odom = None
    # Start timer
    timer = cv2.getTickCount()
    results = {
        'camera_pose': [],
        'detection_pixels': []
    }
    for topic, msg, t in rosbag_result:
        if topic == ODOM_TOPIC:
            odom = Pose.from_ros(msg.pose.pose) # Get odom
            continue
        
        # color_encoding = 'bgr8'
        encodings = ['mono8', 'mono16', 'bgr8', 'bgra8', 'rgb8', 'rgba8']
        frame = bridge.imgmsg_to_cv2(msg, encodings[4])

        # if odom is None: # Wait for first odometry message
        #     continue

        if bbox is None: # Initialize bounding box
            bbox = cv2.selectROI(frame, False)
            tracker.init(frame, bbox)

        # Update tracker
        ok, bbox = tracker.update(frame)
    
        current_time = cv2.getTickCount()
        # Calculate Frames per second (FPS)
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
            break
 
        # Display tracker type on frame
        cv2.putText(frame, "KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
 
        # Display result
        cv2.imshow("Tracking", frame)
        results['camera_pose'].append(odom)
        results['detection_pixels'].append(centre)

        desired_FPS = 60
        desired_wait_ms = int(1000/desired_FPS)
        # Exit if ESC pressed
        k = cv2.waitKey(desired_wait_ms) & 0xff
        if k == 27 : break

    with open('data.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bag.close()


if __name__ == '__main__':
    main()