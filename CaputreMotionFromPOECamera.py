import cv2
import datetime
import os
from collections import deque
import time

# ====== CONFIGURATION ======
camera_user = "root"
camera_password = "password"
camera_ip = "192.168.1.11"
camera_port = 554
camera_channel = 1
rtsp_url = f"rtsp://{camera_user}:{camera_password}@{camera_ip}:{camera_port}/cam/realmonitor?channel={camera_channel}&subtype=0"

output_dir = r"F:\Video_Recordings"
os.makedirs(output_dir, exist_ok=True)

fps = 20
pre_event_seconds = 10
post_event_seconds = 10
pre_event_buffer = deque(maxlen=fps * pre_event_seconds)

motion_detected = False
post_event_timer = None

# ====== VIDEO CAPTURE ======
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

# ====== MOTION DETECTION SETTINGS ======
motion_threshold = 50000  # Adjust based on sensitivity
last_frame = None
recording = False
video_writer = None

print("Monitoring for motion. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Save frame to pre-event buffer
        pre_event_buffer.append(frame.copy())

        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if last_frame is None:
            last_frame = gray
            continue

        frame_delta = cv2.absdiff(last_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = cv2.countNonZero(thresh)

        last_frame = gray

        if motion_score > motion_threshold:
            if not motion_detected:
                print("Motion detected!")
                motion_detected = True
                post_event_timer = time.time() + post_event_seconds

                # Create output file
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = os.path.join(output_dir, f"{timestamp}_motion_event.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

                # Write pre-event frames
                for pre_frame in pre_event_buffer:
                    video_writer.write(pre_frame)

            # Write current frame
            if video_writer:
                video_writer.write(frame)

            # Reset post-event timer
            post_event_timer = time.time() + post_event_seconds

        elif motion_detected:
            if time.time() < post_event_timer:
                if video_writer:
                    video_writer.write(frame)
            else:
                print("Motion ended. Saving video.")
                motion_detected = False
                pre_event_buffer.clear()

                if video_writer:
                    video_writer.release()
                    video_writer = None

except KeyboardInterrupt:
    print("\nStopped by user.")

# Cleanup
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
print("Resources released.")
