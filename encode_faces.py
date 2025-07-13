# Import required libraries for file operations, computer vision, and face recognition
import os  # For working with files and folders
import cv2  # OpenCV: for loading images and using the YuNet model
import face_recognition  # For encoding (converting faces into numerical format)
import pickle  # For saving the encodings to a file

# --- Path configurations ---
dataset_path = 'dataset'  # Folder containing subfolders for each person (each has face images)
encoding_path = 'encodings/face_encodings.pkl'  # File where all face encodings will be saved
thumbnail_dir = 'static/thumbnails'  # Where to store cropped face thumbnails for web display
yunet_model_path = 'models/face_detection_yunet_2023mar.onnx'  # Path to the YuNet ONNX face detector model

# Create the thumbnails folder if it doesn't already exist
os.makedirs(thumbnail_dir, exist_ok=True)

# Print starting message to show the process is beginning
print("🔍 Generating face encodings using YuNet face detector...\n")

# --- Function to detect faces using YuNet ---
def detect_faces_yunet(image, detector, conf_threshold=0.9):
    '''
    This function uses the YuNet AI model to find faces in an image.
    
    Think of it like a smart assistant that can point to where faces are located
    in a photo and tell you how confident it is about each detection.
    
    The function returns the coordinates (positions) of detected faces in a specific
    format that the face_recognition library can understand.
    
    Parameters:
    - image: The photo to analyze
    - detector: The YuNet AI model 
    - conf_threshold: Minimum confidence (0.9 = 90%) to accept a face detection
    '''
    
    # Get the height and width of the image
    h, w = image.shape[:2]
    
    # Tell the YuNet detector what size image we're working with
    detector.setInputSize((w, h))
    
    # Use AI to detect faces in the image
    retval, faces = detector.detect(image)
    
    # Create an empty list to store face coordinates
    boxes = []

    # Check if any faces were found
    if faces is not None and len(faces) > 0:
        # Look at each detected face
        for face in faces:
            # Extract face information: position and confidence
            x, y, w_box, h_box, score = face[:5]
            # x, y = top-left corner coordinates
            # w_box, h_box = width and height of face box
            # score = confidence level (0-1)
            
            # Only keep faces with high confidence
            if score >= conf_threshold:
                '''
                Convert coordinates from YuNet format to face_recognition format:
                - YuNet uses: (x, y, width, height) 
                - face_recognition uses: (top, right, bottom, left)
                '''
                top = int(y)  # Top edge of face
                right = int(x + w_box)  # Right edge of face
                bottom = int(y + h_box)  # Bottom edge of face
                left = int(x)  # Left edge of face
                
                # Add this face's coordinates to our list
                boxes.append((top, right, bottom, left))
    
    return boxes  # Return list of face coordinates

# --- Load YuNet model ---
# Check if the AI model file exists
if not os.path.isfile(yunet_model_path):
    raise FileNotFoundError(f"YuNet model not found at {yunet_model_path}")

# Initialize YuNet face detector
'''
This creates our AI-powered face detector using the YuNet model.
Think of it as training a digital assistant to recognize faces in photos.
'''
detector = cv2.FaceDetectorYN_create(
    yunet_model_path,      # Path to the AI model file
    "",                    # Empty config path (not needed for ONNX format)
    (320, 320),            # Initial input size (will be updated per image)
    score_threshold=0.9,   # Confidence threshold (90%) for accepting a face
    nms_threshold=0.3,     # Threshold to suppress overlapping detections
    top_k=5000             # Maximum number of faces to detect in one image
)

# --- Lists to store encodings and corresponding names ---
known_encodings = []  # List to store encoded face vectors (numerical representations)
known_names = []      # List to store labels corresponding to each encoding

'''
These lists work together like a phone book:
- known_encodings contains the "phone numbers" (mathematical representation of faces)
- known_names contains the "names" (who each face belongs to)
'''

# --- Loop through each person in the dataset ---
# Look at each folder in the dataset directory
for person_name in os.listdir(dataset_path):
    # Create full path to this person's folder
    person_folder = os.path.join(dataset_path, person_name)
    
    # Skip if this isn't actually a folder
    if not os.path.isdir(person_folder):
        continue

    # Print progress message
    print(f"📂 Processing '{person_name}'...")
    
    # Flag to ensure only one thumbnail is saved per person
    thumbnail_saved = False

    # Loop through each image of that person
    for image_name in os.listdir(person_folder):
        # Create full path to this image file
        image_path = os.path.join(person_folder, image_name)
        
        # Load the image from disk
        image = cv2.imread(image_path)

        # Check if image loaded successfully
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            continue  # Skip this image and try the next one

        # Convert image from BGR (Blue-Green-Red) to RGB (Red-Green-Blue)
        # This is needed because face_recognition expects RGB format
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use our YuNet detector to find faces in this image
        face_locations = detect_faces_yunet(image, detector)

        # Check if any faces were found
        if not face_locations:
            print(f"⚠️ No face detected in: {image_name}")
            continue  # Skip this image if no faces found

        # Get face encodings using the detected face boxes
        '''
        This is where the magic happens! The face_recognition library converts
        each face into a list of 128 numbers that mathematically represent
        the unique features of that face.
        
        Think of it like creating a unique fingerprint for each face.
        '''
        encodings = face_recognition.face_encodings(rgb, face_locations)

        # Check if encoding was successful
        if not encodings:
            print(f"⚠️ Face not encoded in: {image_name}")
            continue  # Skip if encoding failed

        # Save encoding and name for each detected face
        for encoding, (top, right, bottom, left) in zip(encodings, face_locations):
            '''
            For each face found in the image:
            1. Add its mathematical representation to known_encodings
            2. Add the person's name to known_names
            3. Save a thumbnail image for the web interface
            '''
            
            # Add the face encoding to our master list
            known_encodings.append(encoding)
            
            # Add the person's name to our master list
            known_names.append(person_name)

            # Save only the first thumbnail per person for UI display
            if not thumbnail_saved:
                # Crop just the face area from the full image
                face_crop = image[top:bottom, left:right]
                
                # Create path for thumbnail file
                thumbnail_path = os.path.join(thumbnail_dir, f"{person_name}.jpg")
                
                # Save the cropped face image as a thumbnail
                cv2.imwrite(thumbnail_path, face_crop)
                
                print(f"🖼️ Thumbnail saved: {thumbnail_path}")
                
                # Mark that thumbnail has been saved for this person
                thumbnail_saved = True

        # Print success message for this image
        print(f"✅ Encoded {len(encodings)} face(s) from: {image_name}")

# --- Summary ---
# Print final statistics about the encoding process
print(f"\n💾 Total faces encoded: {len(known_encodings)}")
print(f"🧑‍🤝‍🧑 Unique persons encoded: {len(set(known_names))}")

# --- Save all encodings to a pickle file ---
'''
Now we save all the face encodings and names to a file so they can be used later.
Think of this like saving a phone book to disk so you can look up numbers later.

The pickle format is Python's way of saving complex data structures to files.
'''

# Create the encodings directory if it doesn't exist
os.makedirs("encodings", exist_ok=True)

# Open the file for writing in binary mode
with open(encoding_path, 'wb') as f:
    # Save both encodings and names as a dictionary
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

# Print success message
print("\n[✅] Face encodings saved to:", encoding_path)