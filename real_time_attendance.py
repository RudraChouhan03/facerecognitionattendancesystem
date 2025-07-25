import cv2  # Import OpenCV library for image and video processing
import face_recognition  # Import face_recognition for facial recognition tasks
import pickle  # Import pickle for loading saved data (face encodings)
import os  # Import os for file and directory operations
import csv  # Import csv for reading and writing CSV files
import numpy as np  # Import numpy for numerical operations
from datetime import datetime  # Import datetime to work with dates and times
from database.database_utils import log_attendance  # Import custom function to log attendance in the database
from spoof_utils import LivenessDetector  # Import liveness detection class to prevent spoofing
'''
Configuration section - These are the main settings for the attendance system.
Think of these as the "control panel" for how the system behaves.
'''
RE_LOG_GAP = 60*2  # Minimum seconds between logging the same person's attendance again
MIN_CONFIDENCE = 0.35  # Threshold for face recognition confidence (lower is stricter)
CSV_FILE = 'attendance_log.csv'  # Path to the CSV file where attendance is logged
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx" # Path to the YuNet face detection model file
ENABLE_LIVENESS_DETECTION = True  # Whether to check if the detected face is real (not a photo/video)
LIVENESS_THRESHOLD = 0.35  # Threshold for deciding if a face is real or spoofed
DEBUG_LIVENESS = True  # Whether to print debug info about liveness detection
'''
Loading face encodings section:
This part loads the saved face data (encodings) that represent each person's face.
'''
try: # Load face encodings (the unique numbers representing each known person's face)
    with open('encodings/face_encodings.pkl', 'rb') as f:  # Open the file with saved face encodings
        data = pickle.load(f)  # Load the data from the file
    print(f"[INFO] Loaded {len(data['encodings'])} face encodings.")  # Print how many faces are known
    print(f"[INFO] Known names: {set(data['names'])}")  # Print the names of known people
except Exception as e:
    print(f"[ERROR] Failed to load encodings: {e}")  # If loading fails, print error and exit
    exit(1)
'''
YuNet face detector initialization:
YuNet is a fast and accurate face detection model that finds faces in images.
It needs to know the size of the video frames to work properly.
'''
# Load YuNet face detector
try:
    # Use a dummy frame to get initial size for YuNet
    dummy_cap = cv2.VideoCapture(0)  # Open the default camera to get a sample frame
    ret, dummy_frame = dummy_cap.read()  # Read a frame from the camera
    dummy_cap.release()  # Release the camera
    if not ret or dummy_frame is None:  # If failed to get a frame
        raise Exception("No camera available to get frame shape for YuNet.")
    h, w = dummy_frame.shape[:2]  # Get height and width of the frame
    yunet = cv2.FaceDetectorYN_create(
        YUNET_MODEL_PATH, "", (w, h), score_threshold=0.9, nms_threshold=0.3, top_k=5000
    )  # Initialize YuNet face detector with the model and frame size
    print("[INFO] YuNet face detector loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YuNet model: {e}")  # If loading fails, print error
    yunet = None  # Set yunet to None to indicate failure
'''
Camera initialization function:
This function tries to find and connect to any available camera.
It checks different camera sources like webcam, USB camera, or CCTV.
'''
# Camera initialization (DroidCam, USB, or CCTV)
def initialize_camera():
    '''
    This function tries to connect to available cameras.
    It checks multiple camera indices and returns the first working camera.
    '''
    camera_indices = [0, 1, 2, 3, 4] # List of camera indices to try (0 is usually the laptop webcam)
    for idx in camera_indices:
        print(f"[INFO] Trying camera index {idx}...")  # Print which camera index is being tried
        cap = cv2.VideoCapture(idx)  # Try to open the camera
        if cap.isOpened():  # If camera opened successfully
            ret, frame = cap.read()  # Try to read a frame
            if ret and frame is not None:  # If frame is valid
                print(f"[INFO] Successfully connected to camera {idx}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height
                cap.set(cv2.CAP_PROP_FPS, 30)  # Set frames per second
                return cap  # Return the working camera
            else:
                cap.release()  # Release if frame not valid
        else:
            cap.release()  # Release if camera not opened
    print("[ERROR] No camera found. Please check your DroidCam/CCTV connection.")  # If no camera found
    return None  # Return None if all attempts fail
cap = initialize_camera()  # Initialize the camera
if cap is None:
    exit(1)  # Exit if camera not found
'''
Liveness detector initialization:
This sets up the system that checks if a detected face is real or fake.
It prevents people from using photos or videos to trick the system.
'''
liveness_detector = None # Initialize liveness detector (checks if face is real or a photo)
if ENABLE_LIVENESS_DETECTION:
    try:
        liveness_detector = LivenessDetector(model_path="models/modelrgb.onnx", threshold=LIVENESS_THRESHOLD)
        print(f"[INFO] Liveness detector initialized with threshold: {LIVENESS_THRESHOLD}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize liveness detector: {e}")
        ENABLE_LIVENESS_DETECTION = False  # Disable liveness detection if initialization fails
'''
Data tracking variables:
These variables keep track of important information during the program's execution.
'''
last_logged = {}  # Dictionary to keep track of last attendance log time for each person
daily_attendance_status = {}  # Dictionary to track daily attendance status
'''
Function to check daily attendance status:
This function looks at the CSV file to see if someone has already marked
their entry or exit for today. This prevents duplicate entries.
'''
def get_daily_attendance_status(name, date_str):
    '''
    Checks the CSV file to see if a person has already marked entry or exit for the given date.
    Returns a dictionary with status and times.
    '''
    try:
        if not os.path.exists(CSV_FILE):
            return {"has_entry": False, "has_exit": False, "entry_time": None, "exit_time": None}
        with open(CSV_FILE, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return {"has_entry": False, "has_exit": False, "entry_time": None, "exit_time": None}
        for row in rows[1:]:
            if len(row) >= 2 and row[0] == name and row[1] == date_str:
                has_entry = len(row) > 2 and row[2].strip() != ""
                has_exit = len(row) > 3 and row[3].strip() != ""
                entry_time = row[2] if has_entry else None
                exit_time = row[3] if has_exit else None
                return {"has_entry": has_entry, "has_exit": has_exit, "entry_time": entry_time, "exit_time": exit_time}
        return {"has_entry": False, "has_exit": False, "entry_time": None, "exit_time": None}
    except Exception as e:
        print(f"[ERROR] Failed to check attendance status: {e}")
        return {"has_entry": False, "has_exit": False, "entry_time": None, "exit_time": None}
'''
Face detection function using YuNet:
This function takes a video frame and finds all the faces in it.
It returns the locations of the faces as rectangular boxes.
'''
def detect_faces_yunet(frame):
    '''
    Detects faces in the given frame using the YuNet model.
    Returns a list of bounding boxes for each detected face.
    '''
    try:
        h, w = frame.shape[:2]
        yunet.setInputSize((w, h))  # Set input size for the model
        retval, faces = yunet.detect(frame)  # Detect faces
        boxes = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                x, y, w_box, h_box, score = face[:5]
                if score >= 0.9:  # Only use detections with high confidence
                    startX, startY = int(x), int(y)
                    endX, endY = int(x + w_box), int(y + h_box)
                    boxes.append([startX, startY, endX, endY])
        return boxes
    except Exception as e:
        print(f"[ERROR] YuNet face detection failed: {e}")
        return []
'''
Face preprocessing for liveness detection:
This function prepares a face image for liveness checking.
It makes sure the face is large enough for reliable detection.
'''
def preprocess_face_for_liveness(face_img):
    '''
    Prepares the face image for liveness detection.
    Skips faces that are too small for reliable detection.
    '''
    if face_img is None or face_img.size == 0:
        return None
    try:
        h, w = face_img.shape[:2]
        if h < 32 or w < 32:
            if DEBUG_LIVENESS:
                print(f"[DEBUG] Face too small: {w}x{h}, skipping liveness detection")
            return None
        return face_img
    except Exception as e:
        print(f"[ERROR] Face preprocessing failed: {e}")
        return None
'''
Liveness detection function:
This function checks if a detected face is real or fake (spoofed).
It uses AI to determine if someone is holding up a photo or video.
'''
def perform_liveness_detection(face_img):
    '''
    Checks if the detected face is real or a spoof (photo/video).
    Returns True if spoof detected, False if real or detection fails.
    '''
    if not ENABLE_LIVENESS_DETECTION or liveness_detector is None:
        return False
    try:
        processed_face = preprocess_face_for_liveness(face_img)
        if processed_face is None:
            if DEBUG_LIVENESS:
                print("[DEBUG] Face preprocessing failed, treating as REAL")
            return False
        is_spoof = liveness_detector.is_spoof(processed_face)
        if DEBUG_LIVENESS:
            try:
                confidence = liveness_detector.get_confidence(processed_face)
                print(f"[DEBUG] Liveness confidence: {confidence:.3f} (threshold: {LIVENESS_THRESHOLD})")
            except:
                print(f"[DEBUG] Liveness result: {'SPOOF' if is_spoof else 'REAL'}")
        return is_spoof
    except Exception as e:
        print(f"[ERROR] Liveness detection failed: {e}")
        if DEBUG_LIVENESS:
            print("[DEBUG] Treating as REAL due to detection failure")
        return False
'''
Main program loop:
This is the heart of the attendance system. It continuously:
1. Captures video frames from the camera
2. Detects faces in each frame
3. Recognizes who each face belongs to
4. Checks if the face is real (not a photo)
5. Logs attendance if a known person is detected
6. Displays the results on screen
'''
# Main loop for real-time attendance
frame_count = 0  # Counts the number of frames processed
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break
    frame_count += 1  # Increment frame count
    boxes = detect_faces_yunet(frame)
    if not boxes:
        cv2.putText(frame, f"Frame: {frame_count} - No faces detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Show message if no faces found
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for face_recognition
        face_locations = [(startY, endX, endY, startX) for (startX, startY, endX, endY) in boxes]  # Convert box format
        try:
            face_encodings = face_recognition.face_encodings(rgb, face_locations)  # Get face encodings for detected faces
        except Exception as e:
            print(f"[ERROR] Face encoding failed: {e}")
            face_encodings = []
        now = datetime.now()  # Get current date and time
        date_str = now.strftime('%Y-%m-%d')  # Format date as string
        time_str = now.strftime('%H:%M:%S')  # Format time as string
        '''
        Process each detected face:
        For each face found in the frame, the system will:
        1. Check if it's a real face (not a photo)
        2. Try to recognize who it is
        3. Draw a colored box around the face
        4. Log attendance if recognized
        '''
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            name = "Unknown"  # Default name if face not recognized
            confidence_text = ""  # Text to show confidence score
            box_color = (0, 0, 255)  # Red box for unknown or spoof
            face_img = frame[startY:endY, startX:endX]  # Crop the face from the frame
            # Check if the face is real or fake
            is_spoof = perform_liveness_detection(face_img)  # Check if face is real or spoofed
            if is_spoof:
                box_color = (0, 0, 255)  # Red box for spoof
                cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
                cv2.putText(frame, "SPOOF DETECTED", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f"[WARNING] Spoof detected for face at ({startX}, {startY})")
                continue  # Skip further processing for spoofed face
            # Try to recognize the face
            if i < len(face_encodings) and len(data["encodings"]) > 0:
                encoding = face_encodings[i]  # Get encoding for current face
                distances = face_recognition.face_distance(data["encodings"], encoding)  # Compare with known encodings
                if len(distances) > 0:
                    min_distance = min(distances)  # Get minimum distance (best match)
                    best_match_index = distances.argmin()  # Get index of best match
                    if min_distance < MIN_CONFIDENCE:
                        name = data["names"][best_match_index]  # Recognized person's name
                        confidence_text = f" ({min_distance:.2f})"  # Show confidence score
                        box_color = (0, 255, 0)  # Green box for recognized face
                        print(f"[INFO] Recognized: {name} (Distance: {min_distance:.2f})")
                        # Log attendance only if enough time has passed since last log
                        # Throttle logging to avoid duplicate logs within RE_LOG_GAP seconds
                        if name not in last_logged or (now - last_logged[name]).total_seconds() > RE_LOG_GAP:
                            try:
                                # NEW LOGIC: Log all appearances to CSV, first/last to DB
                                log_attendance(name, date_str, time_str)  # Log attendance
                                last_logged[name] = now  # Update last logged time
                            except Exception as e:
                                print(f"[ERROR] Attendance logging failed: {e}")
                    else:
                        box_color = (0, 255, 255)  # Yellow box for low confidence
                        confidence_text = f" ({min_distance:.2f})"
                        print(f"[INFO] Unknown or low confidence (Min Distance: {min_distance:.2f})")
            # Draw the face box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)  # Draw rectangle around face
            label = f"{name}{confidence_text}"  # Label to display
            text_color = (255, 255, 255)  # White text 
            cv2.putText(frame, label, (startX, endY - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)  # Show name below the face
    '''
    Display information on the video frame:
    This section adds text overlays to show system status and instructions.
    '''
    if ENABLE_LIVENESS_DETECTION:
        cv2.putText(frame, f"Liveness: ON (Threshold: {LIVENESS_THRESHOLD})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Show liveness status
    else:
        cv2.putText(frame, "Liveness: OFF", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "Press 'q' to quit, 'l' to toggle liveness", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Show controls
    cv2.imshow("Face Recognition Attendance", frame)  # Display the frame
    '''
    Handle keyboard input:
    This section checks for key presses to control the program.
    '''
    key = cv2.waitKey(1) & 0xFF  # Wait for key press
    if key == ord('q'):
        print("[INFO] Quitting attendance tracking.")
        break  # Exit the loop if 'q' is pressed
    elif key == ord('l'):
        ENABLE_LIVENESS_DETECTION = not ENABLE_LIVENESS_DETECTION  # Toggle liveness detection
        print(f"[INFO] Liveness detection {'ENABLED' if ENABLE_LIVENESS_DETECTION else 'DISABLED'}")
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
print("[INFO] Application closed successfully.")  # Print exit message