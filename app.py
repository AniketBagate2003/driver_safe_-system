import math
from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# ------------------------------------------------------------------
# 1. AI Model Initialization
# We use Ultralytics YOLOv8 Pose model to track human skeletal keypoints.
# This avoids broken dependencies and tracks exactly where the user's hand is!
# ------------------------------------------------------------------
model = YOLO("yolov8n-pose.pt") 

# Global flag to track if the driver is currently doing the smoking gesture
SMOKING_DETECTED = False

def calculate_distance(point1, point2):
    # Calculate Euclidean distance between two pixel coordinates
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def gen_frames():
    global SMOKING_DETECTED
    
    # Intialize camera
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Mirrors the camera frame to act like a typical front-facing selfie webcam
        frame = cv2.flip(frame, 1)
        
        # Reset smoking detection flag at start of frame
        smoking_action_this_frame = False

        # Step 1: Run AI Pose Estimation
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Step 2: Extract Skeletal Keypoints
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Keypoint Map in YOLOv8 Pose:
            # 0: Nose (Proxy for mouth/face)
            # 9: Left Wrist (Proxy for Hand holding cigarette)
            # 10: Right Wrist
            
            if len(keypoints) >= 11:
                nose = keypoints[0]
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]
                
                # Verify coordinates are valid (model detects the parts)
                if nose[0] > 0 and nose[1] > 0:
                    dist_left = calculate_distance(nose, left_wrist) if left_wrist[0] > 0 else 9999
                    dist_right = calculate_distance(nose, right_wrist) if right_wrist[0] > 0 else 9999
                    
                    # THRESHOLD TEST: 
                    # We increased the distance to 200 pixels because the 'wrist' keypoint 
                    # is lower than the fingertips. This ensures it triggers reliably!
                    if dist_left < 200 or dist_right < 200:
                        smoking_action_this_frame = True

        # Update global flag thread-safely
        SMOKING_DETECTED = smoking_action_this_frame
        
        # Output visual text onto the actual frame window
        if smoking_action_this_frame:
            cv2.putText(annotated_frame, "SMOKING GESTURE DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Convert frame into streamable bytes
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        # Send chunks to the browser via MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    # Endpiont that javascript pings continuously to check the Global flag
    global SMOKING_DETECTED
    return jsonify({"smoking": SMOKING_DETECTED})

if __name__ == "__main__":
    # Multi-threaded keeps the video streaming from blocking '/status' pings!
    app.run(debug=False, threaded=True, host="127.0.0.1", port=5000)
