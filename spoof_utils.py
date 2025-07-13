# Import required libraries for computer vision, machine learning, and file operations
import cv2  # OpenCV library for computer vision tasks like image processing
import numpy as np  # NumPy for numerical operations and array handling
import onnxruntime as ort  # ONNX Runtime for running machine learning models
import os  # Operating system interface for file and directory operations

class LivenessDetector:
    '''
    This class is responsible for detecting if a face in an image is real (from a live person) 
    or fake (from a photo/video). This prevents people from fooling the system by showing 
    a picture of someone else to the camera.
    
    Think of it like a security guard who can tell the difference between a real person 
    and someone holding up a photo.
    '''
    
    def __init__(self, model_path, threshold=0.5):
        # Store the confidence threshold (0.5 means 50% confidence needed to consider face as real)
        self.threshold = threshold
        
        # Flag to track if the AI model loaded successfully
        self.model_loaded = False
        
        # Check if the AI model file exists at the specified path
        if not os.path.exists(model_path):
            print(f"[WARNING] ONNX model not found at {model_path}")
            print("[INFO] Liveness detection will be disabled")
            return  # Exit early if model file doesn't exist
        
        try:
            '''
            Try to load the ONNX model (a type of AI model format)
            This model has been trained to distinguish between real faces and fake ones
            '''
            # Create an inference session with the model, using CPU for processing
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            # Get the name of the input layer of the neural network
            self.input_name = self.session.get_inputs()[0].name
            
            # Get the expected input shape (dimensions) that the model expects
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Mark that model loaded successfully
            self.model_loaded = True
            
            # Print success messages with model details
            print(f"[INFO] ONNX model loaded successfully. Input shape: {self.input_shape}")
            print(f"[INFO] Input name: {self.input_name}")
            
        except Exception as e:
            # If loading fails, print error and disable liveness detection
            print(f"[ERROR] Failed to load ONNX model: {e}")
            print("[INFO] Liveness detection will be disabled")

    def is_model_available(self):
        '''
        Simple function to check if the liveness detection model is ready to use
        Returns True if model is loaded, False otherwise
        '''
        return self.model_loaded

    def preprocess(self, face_img):
        '''
        This function prepares a face image for the AI model to analyze.
        It's like formatting a document before sending it to a printer - 
        the image needs to be in exactly the right format for the AI to understand it.
        '''
        try:
            # Check if the face image is valid (not empty or corrupted)
            if face_img is None or face_img.size == 0:
                raise ValueError("Invalid face image")
            
            # Get the expected image size from the model (usually 64x64 pixels)
            expected_size = self.input_shape[2] if len(self.input_shape) >= 3 else 64
            
            # Resize the face image to match what the model expects
            img = cv2.resize(face_img, (expected_size, expected_size), interpolation=cv2.INTER_LINEAR)
            
            # Convert color format from BGR (Blue-Green-Red) to RGB (Red-Green-Blue)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to be between 0 and 1 (instead of 0-255)
            img = img.astype(np.float32) / 255.0
            
            # Rearrange dimensions from (height, width, channels) to (channels, height, width)
            img = np.transpose(img, (2, 0, 1))
            
            # Add a batch dimension (needed for neural network processing)
            img = np.expand_dims(img, axis=0)
            
            # Check if the final shape matches what the model expects
            if img.shape != tuple(self.input_shape):
                print(f"[WARNING] Shape mismatch. Expected: {self.input_shape}, Got: {img.shape}")
            
            return img  # Return the processed image
            
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return None  # Return None if preprocessing fails

    def is_spoof(self, face_img):
        '''
        This is the main function that determines if a face is real or fake.
        It analyzes the face image and returns True if it's fake (spoof) or False if it's real.
        
        Think of it as asking: "Is this person trying to trick me with a photo?"
        '''
        
        # If model isn't loaded, skip detection and assume face is real
        if not self.model_loaded:
            print("[INFO] Liveness model not available, skipping spoof detection")
            return False
        
        try:
            # Prepare the face image for the AI model
            input_tensor = self.preprocess(face_img)
            
            # If preprocessing failed, assume it's a spoof for security
            if input_tensor is None:
                print("[WARNING] Preprocessing failed, treating as spoof")
                return True
            
            # Run the AI model to get predictions
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            if len(outputs) > 0:
                output = outputs[0]  # Get the first output
                
                '''
                The model output can be in different formats, so we need to handle each case:
                - Case 1: Two values [spoof_score, live_score]
                - Case 2: One value [live_score]
                - Case 3: Array with multiple values
                '''
                
                # Case 1: Output has 2 values (spoof probability and live probability)
                if len(output.shape) == 2 and output.shape[1] == 2:
                    live_score = output[0][1]  # Probability that face is real
                    spoof_score = output[0][0]  # Probability that face is fake
                
                # Case 2: Output has 1 value (live probability only)
                elif len(output.shape) == 2 and output.shape[1] == 1:
                    live_score = output[0][0]  # Probability that face is real
                    spoof_score = 1.0 - live_score  # Calculate spoof probability
                
                # Case 3: 1D array output
                elif len(output.shape) == 1:
                    if len(output) >= 2:
                        live_score = output[1]  # Second value is live score
                        spoof_score = output[0]  # First value is spoof score
                    else:
                        live_score = output[0]  # Only one value available
                        spoof_score = 1.0 - live_score  # Calculate opposite
                
                # Unexpected output format - try to handle it gracefully
                else:
                    print(f"[WARNING] Unexpected output shape: {output.shape}")
                    live_score = output.flatten()[0] if output.size > 0 else 0.0
                    spoof_score = 1.0 - live_score
                
                # Determine if face is fake: if live_score is below threshold, it's a spoof
                is_spoof = live_score < self.threshold
                
                # Print debug information for troubleshooting
                print(f"[DEBUG] Live score: {live_score:.3f}, Spoof score: {spoof_score:.3f}, Threshold: {self.threshold:.3f}")
                print(f"[DEBUG] Result: {'SPOOF' if is_spoof else 'REAL'}")
                
                return is_spoof  # Return True if fake, False if real
            
            else:
                print("[ERROR] No output from model")
                return True  # If no output, assume it's fake for security
        
        except Exception as e:
            print(f"[ERROR] Liveness detection failed: {e}")
            return True  # If error occurs, assume it's fake for security

    def get_confidence(self, face_img):
        '''
        This function returns how confident the model is that the face is real.
        Returns a number between 0 and 1, where 1 means 100% confident it's real.
        '''
        
        # If model isn't available, return maximum confidence (assume real)
        if not self.model_loaded:
            return 1.0
        
        try:
            # Prepare the image for the model
            input_tensor = self.preprocess(face_img)
            
            # If preprocessing failed, return 0 confidence
            if input_tensor is None:
                return 0.0
            
            # Run the model to get predictions
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            if len(outputs) > 0:
                output = outputs[0]
                
                # Handle different output formats and return the live score
                if len(output.shape) == 2 and output.shape[1] == 2:
                    return float(output[0][1])  # Return live probability
                elif len(output.shape) == 2 and output.shape[1] == 1:
                    return float(output[0][0])  # Return single probability
                else:
                    return float(output.flatten()[0]) if output.size > 0 else 0.0
            
            return 0.0  # Return 0 if no output
        
        except Exception as e:
            print(f"[ERROR] Getting confidence failed: {e}")
            return 0.0  # Return 0 on error

class FaceValidator:
    '''
    This class is responsible for validating uploaded photos to ensure they contain exactly one face.
    It's like a quality checker that makes sure uploaded photos are good enough for the system to use.
    
    It uses YuNet (a face detection model) to find faces in images.
    '''
    
    # Path to the YuNet face detection model file
    YUNET_MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    
    # Standard input size for the YuNet model (320x320 pixels)
    NETWORK_INPUT_SIZE = (320, 320)
    
    # Class variable to store the detector instance (shared across all instances)
    _detector = None

    @classmethod
    def _load_detector(cls):
        '''
        This function loads the YuNet face detection model.
        It's marked as @classmethod which means it belongs to the class, not individual instances.
        '''
        
        # Only load the detector once (if not already loaded)
        if cls._detector is None:
            # Check if the model file exists
            if not os.path.isfile(cls.YUNET_MODEL_PATH):
                raise FileNotFoundError(f"YuNet model not found at {cls.YUNET_MODEL_PATH}")
            
            # Create the face detector using OpenCV's YuNet implementation
            cls._detector = cv2.FaceDetectorYN_create(
                cls.YUNET_MODEL_PATH,  # Path to model file
                "",  # Empty config file path
                cls.NETWORK_INPUT_SIZE,  # Input size for the network
                score_threshold=0.9  # Confidence threshold (90% confidence needed)
            )

    @classmethod
    def validate_uploaded_image(cls, image_path):
        '''
        This function checks if an uploaded image is suitable for face recognition.
        It returns True/False and a message explaining the result.
        
        Requirements:
        - Image must be readable
        - Exactly one face must be detected
        - Face must be clear enough (90% confidence)
        '''
        try:
            # Load the face detector
            cls._load_detector()
            
            # Read the image from the file path
            img = cv2.imread(image_path)
            
            # Check if image was loaded successfully
            if img is None:
                return False, "Could not load image file"
            
            # Get image dimensions (height and width)
            h, w = img.shape[:2]
            
            # Set the input size for the detector to match the image size
            cls._detector.setInputSize((w, h))
            
            # Detect faces in the image
            retval, faces = cls._detector.detect(img)
            
            # Check the results and return appropriate message
            if faces is None or len(faces) == 0:
                return False, "No face detected in the image"
            elif len(faces) > 1:
                return False, "Multiple faces detected. Please upload image with single face"
            else:
                return True, "Face detected successfully"
        
        except Exception as e:
            # Return error message if something goes wrong
            return False, f"Error validating image: {str(e)}"

    @classmethod
    def extract_face_from_image(cls, image_path, output_size=(128, 128)):
        '''
        This function extracts just the face part from an image and resizes it.
        It's like cropping a photo to show only the face, then making it a standard size.
        
        Returns the cropped and resized face image, or None if no face is found.
        '''
        try:
            # Load the face detector
            cls._load_detector()
            
            # Read the image from file
            img = cv2.imread(image_path)
            
            # Check if image was loaded successfully
            if img is None:
                return None
            
            # Get image dimensions
            h, w = img.shape[:2]
            
            # Set detector input size to match image
            cls._detector.setInputSize((w, h))
            
            # Detect faces in the image
            retval, faces = cls._detector.detect(img)
            
            # If at least one face is found, extract the first one
            if faces is not None and len(faces) > 0:
                # Get the coordinates and dimensions of the first face
                x, y, w_box, h_box, score = faces[0][:5]
                
                # Convert coordinates to integers
                x, y, w_box, h_box = map(int, [x, y, w_box, h_box])
                
                # Crop the face from the original image
                face = img[y:y+h_box, x:x+w_box]
                
                # Resize the face to the desired output size
                face_resized = cv2.resize(face, output_size)
                
                return face_resized  # Return the processed face
            
            return None  # Return None if no face found
        
        except Exception as e:
            print(f"[ERROR] Error extracting face: {e}")
            return None  # Return None if error occurs