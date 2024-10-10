import cv2

DATA_FOLDER = './data/'
VIDEO_FOLDER = DATA_FOLDER

def load_video(video_path: str, frame_limit: int = 0) -> list:
    print('Loading video...')
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while cap.isOpened():
        print(f'Frames loaded: {i}', end='\r')
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            break
        frames.append(frame)

        i += 1
        if i >= frame_limit and frame_limit > 0:
            break
    print()
    print('Video loaded')
    cap.release()
    return frames
