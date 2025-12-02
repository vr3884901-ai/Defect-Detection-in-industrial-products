import os
import cv2
import numpy as np
import json
import threading # <--- Import threading for concurrency
import time # <--- Import time for frame rate control
from flask import Flask, render_template, request, redirect, url_for, session, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'neu_defect_classifier.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and classes
model = None
CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches']
TARGET_SIZE = (200, 200) # Standard input size

# Load the Model safely
print("--- LOADING MODEL ---")
try:
    model = load_model(MODEL_PATH)
    print("SUCCESS: Model loaded.")
    print(f"Model Expects Input Shape: {model.input_shape}") 
except Exception as e:
    print(f"ERROR: Could not load model. {e}")
    model = None

def preprocess_image(image, target_size=TARGET_SIZE):
    """Preprocess image for model prediction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# --- Video Stream Threading Class ---
class VideoStream:
    def __init__(self, src=0):
        # Initialize video capture (try 0, then 1)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"Camera {src} failed, trying index {src+1}...")
            self.cap = cv2.VideoCapture(src + 1)
            
        if not self.cap.isOpened():
            print("ERROR: Could not open any camera.")
            self.success = False
        else:
            self.success = True
            self.frame = None
            self.lock = threading.Lock()
            self.stopped = False
            self.prediction_data = {"class": "Waiting for Frame...", "confidence": 0.0}
            
            # Start the thread to read frames
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            
            # Start the thread for prediction (slower task)
            self.pred_thread = threading.Thread(target=self.predict_loop, args=())
            self.pred_thread.daemon = True
            self.pred_thread.start()

    def update(self):
        # Continuously read frames from the camera
        while not self.stopped:
            success, frame = self.cap.read()
            if not success:
                self.stop()
                break
                
            with self.lock:
                self.frame = frame
                
            # Control frame rate slightly to avoid overwhelming the system
            time.sleep(0.03) # ~30 FPS

    def predict_loop(self):
        # Continuously run predictions on the latest frame
        while not self.stopped:
            time.sleep(0.2) # Prediction only runs 5 times per second (much less demanding)
            
            if self.frame is None or model is None:
                continue
                
            try:
                # 1. Get the latest frame safely
                with self.lock:
                    frame_for_pred = self.frame.copy() 
                    
                # 2. Preprocess
                rgb_frame = cv2.cvtColor(frame_for_pred, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                processed = preprocess_image(pil_img, target_size=TARGET_SIZE)
                
                # 3. Predict
                preds = model.predict(processed, verbose=0)
                class_idx = np.argmax(preds[0])
                confidence = np.max(preds[0]) * 100
                
                # 4. Store results
                self.prediction_data = {
                    "class": CLASSES[class_idx],
                    "confidence": float(confidence) 
                }
                
            except Exception as e:
                print(f"Prediction Thread Error: {e}")
                self.prediction_data = {"class": "Model Error", "confidence": 0.0}

    def read_and_encode(self):
        # Read the latest frame and encode it for the video feed
        with self.lock:
            if self.frame is None:
                return None, None
            
            frame_to_display = self.frame.copy()
            pred_data = self.prediction_data.copy()

        # Draw prediction label on the frame
        label = f"{pred_data['class']}: {pred_data['confidence']:.1f}%"
        confidence_val = pred_data['confidence']
        color = (0, 255, 0) if confidence_val > 70 else (0, 0, 255)
        
        frame_to_display = cv2.resize(frame_to_display, (640, 480))
        cv2.putText(frame_to_display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame_to_display)
        frame_bytes = buffer.tobytes()
        
        return frame_bytes, pred_data

    def stop(self):
        self.stopped = True
        self.cap.release()
        
# Instance of the video stream manager
video_stream_manager = None
lock = threading.Lock()

# --- Routes (Unchanged) ---

@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = 'Invalid Credentials.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/detect_image', methods=['GET', 'POST'])
def detect_image():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    prediction_text = "Upload an image to detect"
    uploaded_image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image_url = filepath
            
            try:
                # 1. Open Image
                image = Image.open(filepath)
                
                # 2. Preprocess
                processed_image = preprocess_image(image, target_size=TARGET_SIZE)
                
                # 3. Predict (FIXED: Using predict + argmax)
                preds = model.predict(processed_image)
                class_idx = np.argmax(preds[0]) # Get index of highest probability
                confidence = np.max(preds[0]) * 100
                
                prediction_text = f"Result: {CLASSES[class_idx]} ({confidence:.2f}%)"
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                prediction_text = f"Error: Model prediction failed."

    return render_template('image.html', prediction=prediction_text, image_url=uploaded_image_url)

# --- Live Detection Logic (Modified to use VideoStream class) ---

def generate_frames():
    global video_stream_manager
    
    # Ensure only one instance runs
    with lock:
        if video_stream_manager is None or not video_stream_manager.success:
            video_stream_manager = VideoStream(src=0)
            if not video_stream_manager.success:
                return # Stop if camera couldn't open

    while True:
        frame_bytes, prediction_data = video_stream_manager.read_and_encode()
        
        if frame_bytes is None:
            time.sleep(0.1) # Wait for a frame to be available
            continue

        # Prepare JSON data as a byte string
        json_data = json.dumps(prediction_data).encode('utf-8')

        # FIX: Instead of a custom header, send the JSON payload as a text part, 
        # followed immediately by the image part. This is more robust in browsers.
        yield (
            # 1. Start boundary
            b'--frame\r\n'
            # 2. Text Part: Send JSON payload
            b'Content-Type: application/json\r\n\r\n' + 
            json_data + 
            b'\r\n'
            # 3. Image Part: Send image
            b'Content-Type: image/jpeg\r\n\r\n' + 
            frame_bytes + 
            b'\r\n'
        )

        # Use time.sleep to throttle the feed and prevent CPU spiking
        # 0.05 seconds = 20 FPS (stable for most browsers/systems)
        time.sleep(0.05)


@app.route('/detect_live')
def detect_live():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    # NOTE: We can't use the native <img> tag when we are sending a multi-part stream 
    # that includes JSON *and* the image. We must process the stream entirely in JS.
    # The <img> tag will be kept but the JS will handle the image update.
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Set up global stream manager before running the app
    app.run(debug=True, port=5000, threaded=False)
    # The VideoStream class now handles threading internally.