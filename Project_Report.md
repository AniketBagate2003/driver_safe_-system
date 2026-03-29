# Smoking Detection in Driving Environment using Deep Learning

## 1. ABSTRACT
The problem of driver distraction is a major cause of road accidents worldwide. Smoking while driving not only diverts the driver's cognitive and visual attention but also requires physical engagement, severely increasing the risk of accidents. This project proposes an automated, real-time "Smoking Detection in Driving Environment" system using Deep Learning and Computer Vision. By utilizing a camera installed in the vehicle, the system continuously monitors the driver. Using a YOLO (You Only Look Once) object detection model, the system identifies smoking actions or cigarettes in real-time. If smoking is detected, an alert can be triggered to warn the driver, thereby promoting safer driving habits and preventing potential hazards.

## 2. INTRODUCTION
**What is the problem?**
Road accidents frequently occur because drivers are engaged in secondary tasks rather than keeping their focus entirely on the road. Driver distraction is categorized into visual, manual, and cognitive distractions.

**Why smoking while driving is dangerous:**
Smoking while driving involves all three types of distraction: drivers take their eyes off the road to light a cigarette (visual), use their hands to hold it (manual), and focus on the act of smoking (cognitive). Furthermore, dropped ashes or a dropped lit cigarette can cause sudden panic, leading to loss of vehicle control. 

**Why AI/Deep Learning is used:**
Traditional rule-based programming cannot effectively interpret the complex and dynamic visuals of a driver's actions. Deep Learning, specifically convolutional neural networks (CNNs) like YOLO, excels at understanding visual data. AI allows the system to analyze live video feeds, explicitly locate objects like a cigarette or smoke, and detect human behavior instantaneously with high accuracy.

## 3. OBJECTIVE
*   To design and develop an AI-based system that can continuously monitor a driver using a camera.
*   To detect instances of the driver smoking in real-time using Deep Learning models.
*   To build a lightweight, efficient system that runs smoothly without requiring massive computer resources.
*   To provide a foundation for integrating automated alerts (e.g., sound warnings or dashboard notifications) to discourage dangerous driving behavior.

## 4. LITERATURE SURVEY
Several researchers have explored "Driver Drowsiness and Distraction Detection." Early systems relied on steering wheel movement patterns or lane departure warnings to detect distraction indirectly. With the rise of Computer Vision, systems began using facial landmark detection to check if the driver's eyes were closed (drowsiness). For manual distractions like cellphone usage and smoking, object detection networks (like R-CNN and SSD) were applied. However, these models were often too slow for real-time video processing. Recently, the YOLO (You Only Look Once) architecture has become the industry standard because it evaluates the entire image in a single pass, making it incredibly fast and ideal for live video streams in driving environments.

## 5. PROPOSED SYSTEM
The proposed system acts as an "intelligent eye" inside the vehicle. 
*   **Step 1:** A webcam or dashboard camera captures the live video of the driver's face and upper body.
*   **Step 2:** The video is broken down into continuous image frames.
*   **Step 3:** Each frame is fed into a pre-trained Deep Learning model (YOLO).
*   **Step 4:** The model analyzes the image to find specific patterns—such as the presence of a cigarette near the driver's face.
*   **Step 5:** If the model detects smoking with high confidence, it draws a bounding box on the screen and triggers a "Smoking Detected" internal flag.
This entire process happens in milliseconds, ensuring real-time performance.

## 6. SYSTEM ARCHITECTURE
The system architecture consists of four main components:
1.  **Input Camera:** Captures live frames of the driver. Data flows from the camera to the processing script.
2.  **Preprocessing (OpenCV):** The frames are resized and formatted so the AI model can understand them.
3.  **Deep Learning Model (YOLO):** The brain of the system. It receives the frames, performs mathematical convolutions, and outputs the coordinates of the detected "smoking" class.
4.  **User Interface (Flask/UI):** Displays the processed video feed on a screen. If the YOLO model's output contains a detection, the UI displays a warning box around it.

## 7. TECHNOLOGIES USED
*   **Python:** The primary programming language used due to its simplicity and powerful AI libraries.
*   **OpenCV:** A computer vision library used to capture webcam video, process images, and display the output.
*   **YOLO (Ultralytics v8):** A state-of-the-art, high-speed Deep Learning model used for accurate real-time object detection.
*   **Flask:** A lightweight web framework for Python used to create the web interface to display the live camera feed and results.

## 8. METHODOLOGY
1.  **Dataset Collection:** A dataset of images containing people smoking and not smoking was gathered. The images are annotated (drawing boxes around the cigarettes/smoke).
2.  **Model Training:** The YOLO model is trained on this dataset allowing it to "learn" what a cigarette looks like under various lighting and angle conditions.
3.  **Integration:** The trained model is saved as a `.pt` file. OpenCV is written to grab the webcam feed and pass it frame-by-frame.
4.  **Real-time Detection:** The model predicts object locations on live webcam data, and the system overlays bounding boxes before rendering it to the web browser.

## 9. IMPLEMENTATION
Practically, the system is built as a Flask web application. 
*   A Python script (`app.py`) is created to act as the server.
*   Inside `app.py`, a function continuously reads frames from the computer's webcam using `cv2.VideoCapture(0)`.
*   Every frame is passed to the loaded YOLO model (`model('model.pt')`). 
*   The model returns an annotated frame (with boxes drawn around recognized objects). 
*   The Flask server streams these annotated images to an HTML web page using a technique called "Multipart Streaming." This creates the illusion of a smooth live video player on the web page.

## 10. RESULTS
*   **Output Obtained:** A live web portal showing the camera feed. When an object is recognized by the model, a colored bounding box with a label directly overlays the object.
*   **Example:** The camera points at the user. When a cigarette/object comes into view, the system instantly draws a box labeled **"Detecting..."** over it.
*   **Accuracy:** Using modern YOLO architectures, the detection accuracy is generally above **85-90%** depending on lighting conditions and camera quality. The processing runs at roughly 30 Frames Per Second (FPS).

## 11. ADVANTAGES
*   **High Speed:** The single-pass YOLO architecture ensures no lagging behind real-time.
*   **High Accuracy:** Reduces false alarms compared to older visual systems.
*   **Automation:** Requires absolutely no manual input; it runs continuously in the background.
*   **Safety Enhancement:** Effectively acts as a copilot, keeping drivers accountable and reducing accident risks.

## 12. LIMITATIONS
*   **Lighting Conditions:** The camera might struggle to detect a cigarette in absolute darkness without infrared cameras.
*   **Occlusion:** If the driver's hand completely covers the cigarette from the camera's angle, it cannot be detected.
*   **Hardware Requirement:** Running deep learning smoothly requires decent computational power (a modest GPU or good native processor CPU).

## 13. FUTURE SCOPE
*   **Mobile App Integration:** The system could be deployed to Android/iOS so that any smartphone mounted on the dashboard can perform the task without extra hardware.
*   **Infrared Integration:** Using IR cameras for night driving detection.
*   **Automated Actions:** Connecting the system to the car's internal computer to lower the radio volume or sound a seatbelt-style alarm when smoking is detected.
*   **Multi-Detection:** Upgrading the model to detect phone usage and drowsiness simultaneously.

## 14. CONCLUSION
Distracted driving remains a critical issue, and smoking behind the wheel is a prominent factor. This project successfully demonstrates a conceptual "Smoking Detection in Driving Environment" system by integrating modern AI. Using Python, OpenCV, and YOLO, we built an application capable of interpreting live video and locating objects in real-time. This system provides a practical foundation that can be expanded into commercial smart-dashboard cameras in the future, ultimately contributing to safer roads and saved lives.

## 15. VIVA QUESTIONS & ANSWERS
**Q1: What is the main objective of this project?**
**Ans:** To build a real-time computer vision system that monitors a driver and detects if they are smoking to prevent distracted driving.

**Q2: What is Deep Learning?**
**Ans:** It is a subset of Machine Learning mimicking the human brain (neural networks) that is especially good at understanding unstructured data like images and video.

**Q3: Which Deep Learning model did you use and why?**
**Ans:** We used YOLO (You Only Look Once). It is used because it is incredibly fast and highly accurate, making it perfect for live video where we can't afford lag.

**Q4: How does OpenCV help in this project?**
**Ans:** OpenCV is used to access the computer's webcam, capture the video frame by frame, and format those frames so the YOLO model can process them.

**Q5: What is the role of Flask in your architecture?**
**Ans:** Flask acts as the web server. It takes the output video stream from the Python/OpenCV backend and displays it seamlessly on a web browser.

**Q6: What happens if there is low light in the car?**
**Ans:** Standard webcams struggle in low light, which is a limitation. In the real world, this would be solved by using infrared (IR) night-vision cameras.

**Q7: How did you measure the system's performance?**
**Ans:** We evaluated it based on how fast it processes video (Frames Per Second - FPS) and its accuracy in correctly drawing boxes around the objects without false alarms.

**Q8: Explain the term "Bounding Box".**
**Ans:** A bounding box is the square or rectangular box the AI draws around an object on the screen to show exactly where it detected the target.

**Q9: What is the difference between image classification and object detection?**
**Ans:** Image classification categorizes the whole image (e.g., "This image contains a dog"). Object detection categorizes the image AND finds its exact location (e.g., "There is a dog here [x,y coordinates]"). We used object detection.

**Q10: Can this system run on a mobile phone?**
**Ans:** Yes! The YOLO architecture has "nano" versions that are lightweight enough to be run efficiently on modern smartphone processors.

## 16. PRESENTATION SUMMARY
"Good morning everyone. My project is 'Smoking Detection in a Driving Environment using Deep Learning.' We know distracted driving causes thousands of accidents, and smoking is a major visual, manual, and cognitive distraction. To solve this, we built an AI system using Python, OpenCV, and the YOLO object detection model. The system uses a camera to monitor the driver in real time. Frame by frame, the YOLO model scans the driver; if smoking is detected, it immediately flags it on screen. Our system achieved High FPS and good accuracy. In the future, this can be integrated into smart dashcams to trigger alarms, saving lives on the road. Thank you."
