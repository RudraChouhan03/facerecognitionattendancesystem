'''
This script performs real-time face recognition attendance using a webcam or CCTV.
It loads known face encodings, detects faces in each video frame, recognizes them,
and logs attendance with entry/exit logic into a CSV file and database.

HOW IT WORKS:
1. Opens your camera (webcam or CCTV)
2. Looks at each video frame to find faces
3. Compares found faces with known people in the database
4. If a match is found, records when they entered or left
5. Saves this information in both a CSV file and database
'''

import cv2  # Import OpenCV for video capture and image processing - this handles camera and video
import face_recognition  # Import face_recognition for face detection and recognition - this finds and matches faces
import pickle  # Import pickle to load saved face encodings - this reads the saved face data
import os  # Import os for file system operations - this checks if files exist
import csv  # Import csv for reading and writing CSV files - this handles Excel-like data files
from datetime import datetime  # Import datetime for handling date and time - this gets current time
from database.database_utils import log_attendance  # Import custom attendance logging function - this saves to database
from scipy.spatial import distance  # Import distance for calculating similarity (not directly used)

# Configuration section: set up thresholds and file paths
'''
These are the main settings for the attendance system:
- RE_LOG_GAP: How long to wait before recording the same person again (in seconds)
- MIN_CONFIDENCE: How sure the system needs to be before recognizing someone (0.60 = 60% sure)
- csv_file: The name of the Excel-like file where attendance is saved
- YUNET_MODEL_PATH: Location of the AI model that finds faces in images
'''
RE_LOG_GAP = 60  # Number of seconds to wait before logging the same person again
MIN_CONFIDENCE = 0.60  # Threshold for face recognition match (lower = stricter)
csv_file = 'attendance_log.csv'  # Path to the CSV file for attendance
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"  # Path to the YuNet face detection model

'''
Load known face encodings from a file.
If the file is missing, the script will exit with an error.

WHAT THIS DOES:
- Opens the file that contains all the saved face data of known people
- Loads this data into memory so the system can recognize faces
- If the file doesn't exist, the program stops and tells you to create it first
'''
try:
    with open('encodings/face_encodings.pkl', 'rb') as f:  # Open the encodings file in binary read mode
        data = pickle.load(f)  # Load the pickled data (should contain 'encodings' and 'names')
    print(f"[INFO] Loaded {len(data['encodings'])} face encodings.")  # Print number of encodings loaded
    print(f"[INFO] Known names: {set(data['names'])}")  # Print the set of known names
except FileNotFoundError:  # If the file is not found
    print("[ERROR] Face encodings not found! Please run encode_faces.py first.")  # Print error message
    exit(1)  # Exit the script

# Check if the YuNet model file exists
'''
This checks if the AI model file for face detection is available.
The YuNet model is what actually finds faces in the camera image.
'''
if not os.path.isfile(YUNET_MODEL_PATH):  # If the model file is not found
    print(f"[ERROR] YuNet model not found at {YUNET_MODEL_PATH}")  # Print error
    exit(1)  # Exit script

'''
Try to connect to a camera (webcam or CCTV). Try indices 0, 1, 2.
The first camera that works will be used for video capture.

WHAT THIS DOES:
- Tries to connect to different cameras on your computer
- Index 0 is usually the built-in webcam
- Index 1 and 2 might be external cameras or CCTV
- Uses the first camera that works
'''
cap = None  # Initialize camera variable
for camera_index in [0, 1, 2]:  # Try camera indices 0, 1, and 2
    cap = cv2.VideoCapture(camera_index)  # Attempt to open the camera
    if cap.isOpened():  # If camera opens successfully
        print(f"[INFO] Camera connected at index {camera_index}")  # Print which index worked
        break  # Stop trying other cameras
    cap.release()  # Release the camera if not successful

if not cap or not cap.isOpened():  # If no camera was opened
    print("[ERROR] Could not connect to camera!")  # Print error message
    exit(1)  # Exit script

# Read one frame to get the size for YuNet
'''
This takes one picture from the camera to find out its size (width and height).
The AI model needs to know the image size to work properly.
'''
ret, frame = cap.read()  # Read a single frame from the camera
if not ret:  # If frame was not read successfully
    print("[ERROR] Failed to capture frame from camera")  # Print error
    cap.release()  # Release the camera
    exit(1)  # Exit script
h, w, _ = frame.shape  # Get the height, width, and channels of the frame

'''
Initialize the YuNet face detector using the model file and frame size.
YuNet is a fast and accurate face detection model.

WHAT THIS DOES:
- Sets up the AI system that will find faces in camera images
- Configures it with the right image size and sensitivity settings
- score_threshold: How confident the AI needs to be that it found a face (0.9 = 90% sure)
- nms_threshold: Prevents finding the same face multiple times
- top_k: Maximum number of faces it can find at once
'''
detector = cv2.FaceDetectorYN_create(
    YUNET_MODEL_PATH,  # Path to YuNet model
    "",  # No config file
    (w, h),  # Input size (width, height)
    score_threshold=0.9,  # Only accept detections with high confidence
    nms_threshold=0.3,  # Non-maximum suppression threshold
    top_k=5000  # Maximum number of faces to detect
)

appearance_count = {}  # Dictionary to track how many times each person appeared
last_logged = {}  # Dictionary to track the last time each person was logged

print("[INFO] Attendance tracking started. Press 'q' to quit.")  # Notify user that tracking has started

'''
Detect faces in a frame using YuNet and return bounding boxes in the format required by face_recognition.

WHAT THIS FUNCTION DOES:
- Takes a camera image and finds all the faces in it
- Returns the coordinates of where each face is located
- These coordinates are like invisible rectangles drawn around each face
'''
def detect_faces_yunet(frame, detector):  # Define the function for face detection
    h, w = frame.shape[:2]  # Get frame height and width
    detector.setInputSize((w, h))  # Set detector input size
    retval, faces = detector.detect(frame)  # Detect faces
    boxes = []  # List to store bounding boxes
    if faces is not None and len(faces) > 0:  # If faces are detected
        for face in faces:  # Loop through each detected face
            x, y, w_box, h_box, score = face[:5]  # Get box coordinates and score
            if score >= 0.9:  # Only use boxes with high confidence
                left = int(x)  # Calculate left coordinate
                top = int(y)  # Calculate top coordinate
                right = int(x + w_box)  # Calculate right coordinate
                bottom = int(y + h_box)  # Calculate bottom coordinate
                boxes.append((top, right, bottom, left))  # Append box in correct format
    return boxes  # Return all detected boxes

'''
Update the attendance CSV file for a person.
Handles multiple entries/exits per day by adding new columns if needed.

WHAT THIS FUNCTION DOES:
- Opens the attendance file (like an Excel spreadsheet)
- Finds the row for the person and today's date
- Records their entry or exit time
- If they enter/exit multiple times, it creates new columns automatically
- Saves the updated file
'''
def update_csv(name, date_str, time_str, is_entry=True):  # Define function to update CSV
    rows = []  # List to hold all rows
    if os.path.exists(csv_file):  # If CSV file exists
        with open(csv_file, 'r', newline='') as f:  # Open for reading
            rows = list(csv.reader(f))  # Read all rows

    if not rows:  # If file is empty, add header
        rows.append(["Name", "Date", "Entry1", "Exit1"])  # Add header

    header = rows[0]  # First row is header
    row_index = -1  # Initialize row index

    # Look for existing row for this person and date
    for i, row in enumerate(rows[1:], 1):  # Loop through data rows
        if row[0] == name and row[1] == date_str:  # If name and date match
            row_index = i  # Set row index
            break  # Stop searching

    if row_index == -1:  # If no row found for person and date
        new_row = [name, date_str] + [''] * (len(header) - 2)  # Create new row
        rows.append(new_row)  # Add new row
        row_index = len(rows) - 1  # Set row index

    row = rows[row_index]  # Get the correct row
    while len(row) < len(header):  # If row is shorter than header
        row.append('')  # Add empty cells

    # Find the first empty cell to write the time
    for i in range(2, len(header)):  # Loop through entry/exit columns
        if row[i] == '':  # If cell is empty
            row[i] = time_str  # Write the time
            break  # Stop after writing
    else:  # If all entry/exit columns are filled
        pair_num = (len(header) - 2) // 2 + 1  # Calculate next entry/exit number
        header.extend([f"Entry{pair_num}", f"Exit{pair_num}"])  # Add new columns to header
        row.extend([time_str, ''] if is_entry else ['', time_str])  # Add cells to row

    # Save the updated file
    with open(csv_file, 'w', newline='') as f:  # Open CSV for writing
        csv.writer(f).writerows(rows)  # Write all rows back

'''
Main loop: Reads frames from the camera, detects and recognizes faces, and logs attendance.
Press 'q' to quit the application.

THIS IS THE MAIN PROGRAM THAT RUNS CONTINUOUSLY:
1. Takes pictures from the camera continuously
2. Looks for faces in each picture
3. Tries to recognize who each face belongs to
4. Records when people enter or leave
5. Shows the live camera feed with names above faces
6. Continues until you press 'q' to quit
'''
try:
    while True:  # Loop forever until user quits
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:  # If frame not read successfully
            print("[ERROR] Failed to capture frame from camera")  # Print error
            break  # Exit loop

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        boxes = detect_faces_yunet(frame, detector)  # Detect faces

        face_encodings = face_recognition.face_encodings(rgb, boxes)  # Get face encodings
        print(f"[DEBUG] Detected {len(face_encodings)} face(s)")  # Print how many faces detected

        now = datetime.now()  # Get current time
        date_str = now.strftime('%Y-%m-%d')  # Format date
        time_str = now.strftime('%H:%M')  # Format time

        # Process each detected face
        for encoding, box in zip(face_encodings, boxes):  # Loop through each face
            distances = face_recognition.face_distance(data["encodings"], encoding)  # Calculate distances to known faces
            min_distance = min(distances)  # Find closest match
            best_match_index = distances.argmin()  # Index of closest match

            print(f"[DEBUG] Min Distance: {min_distance:.2f}")  # Print minimum distance
            name = "Unknown"  # Default to unknown

            '''
            Check if the face matches anyone we know.
            If the distance is small enough, it's probably the same person.
            '''
            if min_distance < MIN_CONFIDENCE:  # If match is close enough
                name = data["names"][best_match_index]  # Get matched name
                print(f"[✅] Match: {name} (Distance: {min_distance:.2f})")  # Print match info

                '''
                Only log attendance if enough time has passed since last log.
                This prevents recording the same person multiple times in a short period.
                '''
                if name not in last_logged or (now - last_logged[name]).total_seconds() > RE_LOG_GAP:  # If not logged recently
                    count = appearance_count.get(name, 0) + 1  # Update appearance count
                    appearance_count[name] = count  # Save count
                    is_entry = count % 2 == 1  # Odd = entry, even = exit

                    update_csv(name, date_str, time_str, is_entry)  # Update CSV

                    '''
                    Log to database:
                    - First time seeing someone = entry
                    - Even number of times = exit
                    '''
                    if is_entry and count == 1:  # First entry
                        log_attendance(name, date_str, time_str)  # Log to DB
                    elif not is_entry:  # Exit
                        log_attendance(name, date_str, time_str)  # Log to DB

                    last_logged[name] = now  # Update last log time
                    
                    status = "ENTRY" if is_entry else "EXIT"  # Determine entry/exit
                    print(f"[📝] Logged {status} for {name.title()} at {time_str}")  # Print log info
            else:  # If not recognized
                print(f"[INFO] Unknown or low confidence (Min Distance: {min_distance:.2f})")  # Print info

            '''
            Draw rectangles around faces and write names above them.
            Green rectangle = known person, Red rectangle = unknown person
            '''
            # Draw face box and name
            top, right, bottom, left = box  # Unpack box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  # Draw rectangle
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)  # Draw filled box for name
            cv2.putText(frame, name.title(), (left + 5, bottom - 7),  # Write name
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Use white text

        # Add instructions to the screen
        cv2.putText(frame, "Press 'q' to quit", (10, 30),  # Write quit instruction
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Use white text
        
        cv2.imshow("Face Recognition Attendance", frame)  # Show the frame

        # Check if user wants to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' is pressed
            break  # Exit loop

except KeyboardInterrupt:  # If user presses Ctrl+C
    print("\n[INFO] Interrupted by user")  # Print info
except Exception as e:  # If any error occurs
    print(f"[ERROR] An error occurred: {str(e)}")  # Print error
finally:
    '''
    Clean up resources when the program ends.
    This ensures the camera is properly released and windows are closed.
    '''
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("[INFO] Camera released and windows closed")  # Print info