from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
import os  # For file and directory operations
import cv2  # OpenCV library for camera and image processing
import face_recognition  # Library for face detection and recognition
import pickle  # For saving and loading face encodings data
from datetime import datetime  # For handling date and time operations
import threading  # For handling multiple tasks simultaneously
import shutil  # For file operations like copying and deleting folders
import subprocess  # For running external Python scripts
import sqlite3  # For database operations
import sys  # For system-specific parameters and functions
'''
This line adds the parent directory to the system path so we can import modules from the database folder. 
Think of it as telling Python where to look for our custom database functions.
'''
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.database_utils import log_attendance  # Import our custom attendance logging function
app = Flask(__name__)  # Create Flask application instance
app.secret_key = 'your-secret-key'  # Secret key for session management (change this in production)

DATABASE_PATH = 'database/attendance.db'  # Path to SQLite database file
DATASET_PATH = 'dataset'  # Folder where person photos are stored
ENCODINGS_PATH = 'encodings/face_encodings.pkl'  # File where face encodings are saved
THUMBNAIL_PATH = 'static/thumbnails'  # Folder for thumbnail images
ADMIN_USERNAME = 'rudra'  # Admin username for login
ADMIN_PASSWORD = 'rudra123'  # Admin password for login

# Global variables to manage camera and application state
camera = None  # Will store the camera object
camera_lock = threading.Lock()  # Prevents multiple threads from accessing camera simultaneously
face_encodings_data = None  # Will store all face encodings data
current_mode = 'capture'  # Current application mode (capture or recognition)
'''
Multi-photo capture session variables
These variables control the process of capturing multiple photos
for better face recognition accuracy
'''
capture_session_active = False  # Is photo capture session currently running?
capture_photo_index = 0  # Which photo number are we currently capturing?
capture_total_photos = 5  # Total number of photos to capture per person
# Instructions shown to user for each photo to get different angles
directions = [
    "face front towards camera",
    "turn face slightly to the left",
    "turn face slightly to the right",
    "look up slightly",
    "look down slightly"
]
import csv  # For CSV file operations
def load_face_encodings():
    '''
    This function loads all the saved face encodings from the pickle file.
    Face encodings are mathematical representations of faces that the
    computer can use to recognize people.
    '''
    global face_encodings_data  # Access the global variable
    try:
        # Open the pickle file and load the face encodings data
        with open(ENCODINGS_PATH, 'rb') as f:
            face_encodings_data = pickle.load(f)
        # Print success message with number of encodings loaded
        print(f"[INFO] Loaded {len(face_encodings_data['encodings'])} face encodings.")
    except Exception as e:
        # If loading fails, print error and set data to None
        print("[ERROR] Could not load face encodings:", e)
        face_encodings_data = None
def get_camera():
    '''
    This function gets the camera object. If camera is not already
    initialized, it creates a new camera connection.
    '''
    global camera  # Access the global camera variable
    if camera is None:
        camera = cv2.VideoCapture(0)   # Create camera object (0 means default camera/webcam)
    return camera
def release_camera_resource():
    '''
    This function safely releases the camera so other applications
    can use it. It uses a lock to prevent conflicts.
    '''
    global camera  # Access the global camera variable
    with camera_lock:  # Use lock to prevent other threads from interfering
        if camera is not None:
            camera.release()  # Release the camera
            camera = None  # Set camera variable to None
def generate_frames():
    '''
    This function continuously captures frames from the camera and
    sends them to the web browser for live video streaming.
    It's like a live video feed on a website.
    '''
    global current_mode  # Access the global mode variable
    cam = get_camera()  # Get the camera object
    try:
        while True:  # Keep running forever
            with camera_lock:  # Use lock to safely access camera as with acquires the lock and releases later
                ret, frame = cam.read()  # Capture a frame from camera
            if not ret:  # If frame capture failed
                break  # Exit the loop
            
            # If we're in recognition mode and have face encodings loaded
            if current_mode == 'recognition' and face_encodings_data:
                frame = process_recognition_frame(frame)  # Process frame for face recognition
            
            # Convert frame to JPG format for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            # Yield the frame in the format needed for web streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        release_camera_resource()  # Always release camera when done
def process_recognition_frame(frame):
    '''
    This function processes each camera frame to detect and recognize faces.
    It draws rectangles around detected faces and shows the person's name.
    ''' 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR (Blue-Green-Red) to RGB (Red-Green-Blue) format
    face_locations = face_recognition.face_locations(rgb)  # Find all face locations in the frame
    face_encodings = face_recognition.face_encodings(rgb, face_locations)   # Get face encodings for all detected faces
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):   # Process each detected face
        name = "Unknown"  # Default name if face is not recognized
        if face_encodings_data:  # If we have face encodings data loaded
            distances = face_recognition.face_distance(face_encodings_data["encodings"], encoding)  # Compare current face with all saved faces. facd_encoding_data is a dictionary with names and encodings and we want only encodings.
            if len(distances) > 0 and min(distances) < 0.6:   # If we found matches and the best match is good enough (distance < 0.6)
                best_match = distances.argmin() # finds the index of the smallest value in the distances array.
                name = face_encodings_data["names"][best_match]  # Get the person's name
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)  # Draw a green rectangle around the face
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)  # Write the person's name above the rectangle
    return frame  # Return the processed frame

# --- Multi-photo capture session endpoints ---
@app.route('/start_capture', methods=['POST'])
def start_capture():
    '''
    This function starts a photo capture session for registering a new person.
    It prepares the system to capture multiple photos of the same person.
    '''
    global capture_session_active, capture_photo_index, camera
    if capture_session_active:  # Check if a capture session is already running
        return jsonify({'success': False, 'message': 'Capture session already active'})
    # Start new capture session
    capture_session_active = True
    capture_photo_index = 0  # Start with first photo
    camera = get_camera()  # Initialize camera
    # Send response with first instruction
    return jsonify({
        'success': True, 
        'message': 'Capture session started', 
        'instruction': directions[capture_photo_index], 
        'photo_number': 1
    })
@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    '''
    This function captures a single photo during the capture session.
    It saves the photo and provides instruction for the next photo.
    '''
    global capture_session_active, capture_photo_index, camera
    if not capture_session_active:  # Check if capture session is active
        return jsonify({'success': False, 'message': 'No active capture session'})
    person_name = request.json.get('person_name')   # Get person name from the request
    if not person_name:
        return jsonify({'success': False, 'message': 'Person name required'})
    # Capture frame from camera
    with camera_lock:  # Use lock for thread safety
        ret, frame = camera.read()
    if not ret:  # If frame capture failed
        return jsonify({'success': False, 'message': 'Failed to capture frame'})
    person_folder = os.path.join(DATASET_PATH, person_name)  # Create folder for this person if it doesn't exist
    os.makedirs(person_folder, exist_ok=True)
    photo_num = capture_photo_index + 1
    filename = f"{photo_num}.jpg" # Generate filename for the photo
    filepath = os.path.join(person_folder, filename)
    cv2.imwrite(filepath, frame)  # Save the captured frame as an image file
    capture_photo_index += 1  # Prepare for next photo
    if capture_photo_index >= len(directions):   # Check if all photos are captured
        msg = f"All {len(directions)} photos captured. You can stop capture now."
        instruction = ""
        finished = True
    else:
        msg = f"Captured photo {photo_num}. Next: {directions[capture_photo_index]}"
        instruction = directions[capture_photo_index]
        finished = False
    return jsonify({
        'success': True, 
        'message': msg, 
        'instruction': instruction, 
        'photo_number': photo_num, 
        'finished': finished
    })  # Send response with next instruction
@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    '''
    This function stops the photo capture session and releases the camera.
    It handles both active capture sessions and general camera release.
    '''
    global capture_session_active, camera, capture_photo_index
    # Check if there was an active capture session
    was_capture_active = capture_session_active
    # Always release camera and reset capture session variables
    release_camera_resource()  # Release camera
    capture_session_active = False  # Mark session as inactive
    capture_photo_index = 0  # Reset photo index
    # Return appropriate message based on previous state
    if was_capture_active:
        return jsonify({'success': True, 'message': 'Capture session stopped and camera released'})
    else:
        return jsonify({'success': True, 'message': 'Camera released from video feed'})
@app.route('/login', methods=['GET', 'POST'])
def login():
    '''
    This function handles user login. It shows the login page and
    checks if the entered username and password are correct.
    '''
    if request.method == 'POST':  # If user submitted login form
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:  # Check if credentials match admin credentials
            session['logged_in'] = True  # Mark user as logged in
            return redirect(url_for('dashboard'))  # Redirect to dashboard      
        return render_template('login.html', error='Invalid credentials')  # If credentials are wrong, show error
    return render_template('login.html') # If GET request, just show the login page
@app.route('/logout')
def logout():
    '''
    This function logs out the user by removing the login session.
    '''
    session.pop('logged_in', None)  # Remove login status from session
    return redirect(url_for('login'))  # Redirect to login page
@app.route('/')
def dashboard():
    '''
    This function shows the main dashboard with attendance records.
    It also handles filtering of records by name and date.
    '''
    if 'logged_in' not in session:  # Check if user is logged in
        return redirect(url_for('login'))
    # Get filter parameters from URL
    filter_name = request.args.get('filter_name', '').strip().lower()
    filter_date = request.args.get('filter_date', '').strip()
    # Connect to database and get attendance records
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, first_entry, last_exit FROM daily_attendance ORDER BY date DESC")
    records = cursor.fetchall()  # Get all records
    conn.close()  # Close database connection
    # Convert database records to a more readable format
    attendance_records = [
        {'name': r[0], 'date': r[1], 'entry_time': r[2], 'exit_time': r[3]}
        for r in records
    ]
    # Apply filters to the attendance records. This allows users to search for specific people or dates
    if filter_name:  # Filter by name if provided
        attendance_records = [rec for rec in attendance_records if filter_name in rec['name'].lower()]
    if filter_date:   # Filter by date if provided
        attendance_records = [rec for rec in attendance_records if rec['date'] == filter_date]
    return render_template('index.html', attendance_records=attendance_records)  # Show the dashboard with filtered records
@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    # This function handles photo upload for registering new people. Users can upload photos instead of capturing them with camera.
    if 'logged_in' not in session:   # Check if user is logged in
        return jsonify({'success': False, 'message': 'Not authorized'})
    person_name = request.form.get('person_name')  # Get person name and check if image file is provided
    if not person_name or 'image' not in request.files:
        return jsonify({'success': False, 'message': 'Missing data'})
    person_folder = os.path.join(DATASET_PATH, person_name)  # Create folder for this person
    os.makedirs(person_folder, exist_ok=True)
    existing_photos = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png'))]  # Find existing photos to determine next filename
    next_number = len(existing_photos) + 1
    filename = f"{next_number}.jpg"
    request.files['image'].save(os.path.join(person_folder, filename))  # Save the uploaded image
    return jsonify({'success': True, 'message': f'Photo uploaded successfully for {person_name}'})
@app.route('/generate_encodings', methods=['POST'])
def generate_encodings():
    # This function generates face encodings from all photos in the dataset. Face encodings are mathematical representations that help recognize faces.
    if 'logged_in' not in session:   # Check if user is logged in
        return jsonify({'success': False, 'message': 'Not authorized'})
    try:
        if not os.path.exists('encode_faces.py'):  # Check if the encoding script exists
            return jsonify({'success': False, 'message': 'encode_faces.py not found'})
        subprocess.run([sys.executable, 'encode_faces.py'], check=True)   # Run the encoding script
        load_face_encodings()  # Load the newly generated encodings
        return jsonify({'success': True, 'message': 'Encodings generated successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
@app.route('/test_recognition_result', methods=['POST'])
def test_recognition_result():
    '''
    This function tests face recognition by taking a photo and trying to recognize the person in it. It shows confidence levels.
    '''
    if 'logged_in' not in session:    # Check if user is logged in
        return jsonify({'success': False, 'message': 'Not authorized'})
    load_face_encodings()    # Load face encodings for recognition
    cam = get_camera()  # Capture a frame from camera
    with camera_lock:
        ret, frame = cam.read()
    if not ret:  # If frame capture failed
        release_camera_resource()
        return jsonify({'success': False, 'message': 'Camera error'})
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    results = []  # Store recognition results
    for encoding in face_encodings:  # Process each detected face
        name = "Unknown"
        confidence = 0
        if face_encodings_data:   # If we have face encodings data
            distances = face_recognition.face_distance(face_encodings_data["encodings"], encoding)   # Compare with all known faces
            # If we found a good match
            if len(distances) > 0 and min(distances) < 0.6:
                best_match = distances.argmin()
                name = face_encodings_data["names"][best_match] 
                confidence = round((1 - min(distances)) * 100, 2)   # Calculate confidence percentage
        results.append({'name': name, 'confidence': confidence})
    release_camera_resource()  # Release camera
    return jsonify({'success': True, 'results': results, 'message': 'Recognition test completed.'})
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    '''
    This function manually marks attendance for a person. It records the current date and time in the database.
    '''
    if 'logged_in' not in session: # Check if user is logged in
        return jsonify({'success': False, 'message': 'Not authorized'})
    person_name = request.json.get('person_name')   # Get person name from request
    if not person_name:
        return jsonify({'success': False, 'message': 'Person name required'})
    # Get current date and time
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    try:
        action = log_attendance(person_name, date_str, time_now) # Log attendance in database
        return jsonify({'success': True, 'message': f'Marked {action} for {person_name}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Attendance logging failed: {e}'})
@app.route('/remove_person', methods=['POST'])
def remove_person():
    '''
    This function removes a person from the system by deleting their photos and regenerating face encodings.
    '''
    if 'logged_in' not in session:  # Check if user is logged in
        return jsonify({'success': False, 'message': 'Not authorized'})
    person_name = request.json.get('person_name')  # Get person name from request
    if not person_name:
        return jsonify({'success': False, 'message': 'Person name required'})
    person_folder = os.path.join(DATASET_PATH, person_name) # Check if person's folder exists
    if os.path.exists(person_folder):
        shutil.rmtree(person_folder) # Remove person's photo folder
        thumbnail_path = os.path.join(THUMBNAIL_PATH, f"{person_name}.jpg") # Remove person's thumbnail if it exists
        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)
        if os.path.exists('encode_faces.py'):   # Regenerate face encodings without this person
            try:
                subprocess.run([sys.executable, 'encode_faces.py'], check=True)
                load_face_encodings()  # Reload encodings
            except Exception as e:
                return jsonify({'success': False, 'message': f'Person removed but encoding regeneration failed: {e}'})
        return jsonify({'success': True, 'message': f'Successfully removed {person_name}'})
    return jsonify({'success': False, 'message': 'Person not found'})
@app.route('/start_live_attendance')
def start_live_attendance():
    '''
    This function starts the live attendance mode where the camera continuously looks for faces and can mark attendance automatically.
    '''
    if 'logged_in' not in session: # Check if user is logged in
        return redirect(url_for('login'))
    global current_mode
    current_mode = 'recognition'  # Set mode to recognition
    load_face_encodings()  # Load face encodings
    return render_template('live_recognition.html')   # Show the live recognition page
@app.route('/video_feed')
def video_feed():
    '''
    This function provides the live video feed for the web page. It continuously sends camera frames to the browser.
    '''
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/export_csv')
def export_csv():
    '''
    This function exports all attendance records to a CSV file that can be downloaded and opened in Excel or similar programs.
    '''
    if 'logged_in' not in session:   # Check if user is logged in
        return redirect(url_for('login'))
    try:
        conn = sqlite3.connect(DATABASE_PATH)  # Connect to database
        df = None  
        try:
            import pandas as pd    # Try to use pandas to read data (if available)
            df = pd.read_sql_query("SELECT * FROM daily_attendance ORDER BY date DESC", conn)
        finally:
            conn.close()  # Always close database connection
        if df is None or df.empty:   # Check if we have data to export
            return '''<html><body><h2>Export Error</h2><p>No attendance records found to export.</p><button onclick="window.history.back()">Go Back</button></body></html>'''
        csv_filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"   # Generate unique filename with timestamp
        csv_path = os.path.join('exports', csv_filename)
        abs_csv_path = os.path.abspath(csv_path)
        df.to_csv(abs_csv_path, index=False)  # Save data to CSV file
        from flask import send_file       # Send file to user for download
        return send_file(abs_csv_path, as_attachment=True, download_name=csv_filename)
    except Exception as e:
        return f'''<html><body><h2>Export Error</h2><p>Error exporting CSV: {str(e)}</p><p>Current working directory: {os.getcwd()}</p><button onclick="window.history.back()">Go Back</button></body></html>'''        # If export fails, show error message
@app.route('/set_mode', methods=['POST'])
def set_mode():
    '''
    This function changes the application mode between 'capture' and 'recognition'. Different modes change how the camera processes the video feed.
    '''
    if 'logged_in' not in session:   # Check if user is logged in
        return jsonify({'success': False, 'message': 'Not authorized'})
    global current_mode
    mode = request.json.get('mode')
    if mode in ['capture', 'recognition']:   # Check if mode is valid
        current_mode = mode
        return jsonify({'success': True, 'message': f'Mode set to {mode}'})
    return jsonify({'success': False, 'message': 'Invalid mode'})
@app.route('/release_camera', methods=['POST'])
def release_camera():
    '''
    This function releases the camera so other applications can use it.
    '''
    release_camera_resource()
    return jsonify({'success': True, 'message': 'Camera released'})
@app.route('/popup_capture_faces')
def popup_capture_faces():
    '''
    This function opens a separate window for capturing faces. It runs the capture_faces.py script in a new process.
    '''
    if 'logged_in' not in session:    # Check if user is logged in
        return redirect(url_for('login'))
    try:
        subprocess.Popen([sys.executable, 'capture_faces.py'])   # Start the capture faces script in a new process  
        return 
    except Exception as e:
        return f"<h3>Error launching capture_faces.py: {e}</h3>"
@app.route('/popup_encode_faces')
def popup_encode_faces():
    '''
    This function opens a separate window for encoding faces. It runs the encode_faces.py script in a new process.
    '''
    if 'logged_in' not in session:   # Check if user is logged in
        return redirect(url_for('login'))
    try:
        subprocess.Popen([sys.executable, 'encode_faces.py'])    # Start the encode faces script in a new process
        return
    except Exception as e:
        return f"<h3>Error launching encode_faces.py: {e}</h3>"
@app.route('/popup_real_time_attendance')
def popup_real_time_attendance():
    '''
    This function opens a separate window for real-time attendance. It runs the real_time_attendance.py script in a new process.
    '''
    if 'logged_in' not in session:   # Check if user is logged in
        return redirect(url_for('login'))
    try:
        subprocess.Popen([sys.executable, 'real_time_attendance.py'])     # Start the real-time attendance script in a new process
        # Show alert and close popup
        return '''
        <script>
            alert("Real-Time Attendance window opened. Please use the new window for attendance.");
            window.close();
        </script>
        '''
    except Exception as e:
        return f"<h3>Error launching real_time_attendance.py: {e}</h3>"
'''
Main application entry point. This code runs when the script is executed directly (not imported)
'''
if __name__ == '__main__':
    load_face_encodings()  # Load face encodings when app starts
    app.run(debug=True, threaded=True)  # Start the Flask web server