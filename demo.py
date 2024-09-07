import cv2
import mediapipe as mp

# Initialize MediaPipe hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Initialize MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera")
            break
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame using MediaPipe hands
        try:
            results = hands.process(rgb_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
        
        # Draw hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the output
        cv2.imshow('MediaPipe Hand Recognition', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()