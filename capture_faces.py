# Import required libraries for camera, file operations, speech, and time management
import cv2  # OpenCV for camera access and image processing
import os   # For working with file and directory paths
import time  # Used for delays between instructions
import pyttsx3  # Text-to-speech engine for giving voice instructions
import shutil  # For deleting directories and their contents

# YuNet ONNX model path for face detection
YUNET_MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")

# List of 5 different face poses we want the user to show
POSES = [
    "face front towards camera",           # Pose 1: Looking straight ahead
    "turn face slightly to the left",      # Pose 2: Head turned left
    "turn face slightly to the right",     # Pose 3: Head turned right
    "look up slightly",                    # Pose 4: Looking up
    "look down slightly"                   # Pose 5: Looking down
]

def speak(text):
    '''
    This function converts text to speech using the computer's voice.
    It's like having a digital assistant that can talk to the user.
    
    The function both speaks the text out loud AND prints it on screen,
    so the user can hear AND see the instructions.
    '''
    print(f"[Instruction] {text}")  # Print the instruction text on screen
    engine.say(text)  # Tell the text-to-speech engine what to say
    engine.runAndWait()  # Wait for the speech to finish before continuing

def capture_faces(person_name, camera_index=0):
    '''
    This is the main function that captures 5 face photos of a person.
    
    Think of it like a photo booth session where:
    1. The computer tells you how to pose
    2. You adjust your face position
    3. Press ENTER when ready
    4. The computer saves a cropped picture of just your face
    5. Repeat for 5 different poses
    
    All photos are saved in a folder named after the person.
    '''
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the path where photos will be saved: dataset/person_name/
    dataset_path = os.path.join(script_dir, "dataset", person_name)

    # If folder for this person already exists, delete it to avoid mixing old and new photos
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)  # Delete the entire folder and its contents
    
    # Create a fresh, empty folder for this person's photos
    os.makedirs(dataset_path, exist_ok=True)

    # Initialize the camera (webcam)
    cap = cv2.VideoCapture(camera_index)  # camera_index=0 means the default camera
    
    # Check if camera is working
    if not cap.isOpened():
        print("❌ Error: Webcam not accessible.")
        return  # Exit the function if camera doesn't work

    # Read one frame to get dimensions (width and height)
    ret, frame = cap.read()  # ret=True if successful, frame=the actual image
    if not ret:
        print("❌ Failed to read from camera.")
        cap.release()  # Close the camera
        return  # Exit the function

    # Get frame dimensions
    h, w, _ = frame.shape  # h=height, w=width, _=color channels (ignored)

    # Initialize YuNet face detector
    '''
    YuNet is an AI model that can find faces in images.
    Think of it as a smart system that can point to where faces are located.
    '''
    detector = cv2.FaceDetectorYN_create(
        YUNET_MODEL_PATH,     # Path to the AI model file
        "",                   # No additional config file needed
        (w, h),               # Input size = webcam frame size
        score_threshold=0.9,  # Minimum confidence (90%) to accept a face detection
        nms_threshold=0.3,    # For removing duplicate/overlapping face boxes
        top_k=5000            # Maximum number of faces to detect in one image
    )

    # Welcome message - greet the person and explain what will happen
    speak(f"Hello {person_name}, we will now capture 5 cropped face photos.")

    # Loop through each of the 5 poses
    for i, pose in enumerate(POSES, start=1):
        '''
        This loop runs 5 times, once for each pose.
        
        For each pose, it:
        1. Tells the user what pose to do
        2. Waits for them to adjust their position
        3. Shows live camera feed with face detection
        4. Waits for user to press ENTER
        5. Saves the cropped face image
        '''
        
        # Tell the user what pose to do (e.g., "Please face front towards camera")
        speak(f"Please {pose}")
        
        # Give the user 2 seconds to adjust their position
        time.sleep(2)

        # Keep showing camera feed until user presses ENTER
        while True:
            # Read the current frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame.")
                continue  # Try again if frame reading failed

            # If webcam resolution changes, update the detector input size
            if frame.shape[0] != h or frame.shape[1] != w:
                h, w, _ = frame.shape  # Update stored dimensions
                detector.setInputSize((w, h))  # Tell detector about new size

            # Use AI to detect faces in the current frame
            retval, faces = detector.detect(frame)
            
            # Flag to track if we found a face
            face_found = False

            # Check if any faces were detected
            if faces is not None and len(faces) > 0:
                '''
                If multiple faces are detected, we want to pick the best one.
                Each face detection comes with a confidence score.
                We sort by confidence and pick the highest scoring face.
                '''
                
                # Sort faces by confidence score (highest first)
                faces = sorted(faces, key=lambda x: x[4], reverse=True)
                
                # Get the coordinates of the best face
                x, y, w_box, h_box, score = faces[0][:5]
                # x, y = top-left corner of face box
                # w_box, h_box = width and height of face box
                # score = confidence level
                
                # Convert coordinates to integers (required for drawing)
                x, y, w_box, h_box = map(int, [x, y, w_box, h_box])

                # Draw a green rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                # (0, 255, 0) = green color in BGR format
                # 2 = thickness of the rectangle border
                
                # Mark that we found a face
                face_found = True

            # Show the camera feed with face detection box
            cv2.imshow("Capture - Press ENTER to capture", frame)
            
            # Check if any key was pressed
            key = cv2.waitKey(1)  # Wait 1 millisecond for a key press
            
            # If ENTER key was pressed (key code 13)
            if key == 13:
                # Check if we detected a face
                if not face_found:
                    speak("No face detected. Please adjust your position.")
                    continue  # Go back to camera feed, don't save anything

                # Crop the face region from the full frame
                face_img = frame[y:y + h_box, x:x + w_box]
                # This extracts just the rectangular area containing the face

                # Create the file path where this photo will be saved
                img_path = os.path.join(dataset_path, f"{i}.jpg")
                # Example: dataset/john_doe/1.jpg, dataset/john_doe/2.jpg, etc.
                
                # Save the cropped face image to disk
                cv2.imwrite(img_path, face_img)
                
                # Confirm to user that photo was saved
                speak(f"Captured photo {i}")
                print(f"[Saved] {img_path}")
                
                # Break out of the while loop to move to next pose
                break

        # Small delay before starting the next pose
        time.sleep(1)

    # All 5 photos have been captured successfully
    speak("All 5 face photos captured successfully.")
    
    # Clean up: close camera and windows
    cap.release()  # Stop using the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    '''
    This block only runs when the script is executed directly (not imported as a module).
    
    It's like the "main" function that starts everything when you run the script.
    '''
    
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Set speaking speed to 160 words per minute (comfortable listening speed)
    engine.setProperty('rate', 160)
    
    # Ask the user to enter the person's name
    person_name = input("Enter the name of the person: ").strip()
    # .strip() removes any extra spaces before/after the name
    
    # Check if a name was actually entered
    if person_name:
        # Start the face capture process
        capture_faces(person_name)
    else:
        # Show error if no name was provided
        print("❌ Name cannot be empty.")