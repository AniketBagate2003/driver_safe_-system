# Smoking Detection in Driving Environment

A complete AI-driven vehicle dashboard application built with Python, OpenCV, Flask, and Ultralytics YOLOv8. Includes a modern Glassmorphism web UI.

## How to Run

1. **Install Python**: Ensure you have Python installed on your Windows machine.
2. **Open Terminal**: Navigate to this project folder.
3. **Install Dependencies**: Run the following command:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application**:
   ```bash
   python app.py
   ```
5. **View the Dashboard**: Open your web browser and go to `http://127.0.0.1:5000`

## Academic / Custom Model Note

For the purpose of having a functional plug-and-play demo immediately, this script currently loads the powerful lightweight `yolov8n.pt` base model. This base model will automatically download on the first run and identifies 80 common objects (like a person, cell phones, cups, etc.) to prove the detection pipeline works perfectly.

**To make it specifically detect cigarettes / smoking:**
1. Train a custom YOLOv8 model using an annotated dataset (e.g., download a smoking dataset from Roboflow or Kaggle, and upload to Google Colab for fast training).
2. Save the trained weights file (usually outputted as `best.pt`).
3. Replace `"yolov8n.pt"` on line 13 inside `app.py` with your custom `"best.pt"`.
