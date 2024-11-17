import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
camera = cv2.VideoCapture(0)

# Function to classify ASL letters based on landmarks
def classify_asl_letter(hand_landmarks, image_shape):
    """
    Classifies the hand gesture into an ASL letter based on the positions of landmarks.
    """
    landmarks = hand_landmarks.landmark

    # Normalize landmark positions relative to the wrist
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Relative positions of fingers
    is_thumb_folded = thumb_tip.x < wrist.x if thumb_tip.y > index_tip.y else thumb_tip.x > wrist.x
    is_index_extended = index_tip.y < landmarks[5].y
    is_middle_extended = middle_tip.y < landmarks[9].y
    is_ring_extended = ring_tip.y < landmarks[13].y
    is_pinky_extended = pinky_tip.y < landmarks[17].y

    # ASL letter classification logic
    if all([not is_index_extended, not is_middle_extended, not is_ring_extended, not is_pinky_extended, is_thumb_folded]):
        return "A"
    elif all([is_index_extended, is_middle_extended, not is_ring_extended, not is_pinky_extended, is_thumb_folded]):
        return "B"
    elif all([not is_index_extended, not is_middle_extended, not is_ring_extended, not is_pinky_extended, is_thumb_folded]):
        return "C"
    elif all([is_index_extended, is_middle_extended, not is_ring_extended, not is_pinky_extended]):
        return "D"
    elif all([not is_index_extended, not is_middle_extended, not is_ring_extended, is_pinky_extended, is_thumb_folded]):
        return "E"
    elif all([is_index_extended, not is_middle_extended, not is_ring_extended, is_pinky_extended]):
        return "F"
    elif is_thumb_folded and is_index_extended and not is_middle_extended:
        return "G"
    elif is_thumb_folded and is_index_extended and is_middle_extended:
        return "H"
    elif is_pinky_extended and not is_index_extended:
        return "I"
    elif is_pinky_extended and not is_index_extended and landmarks[20].x < landmarks[17].x:
        return "J"
    elif is_index_extended and is_middle_extended and not is_ring_extended:
        return "K"
    elif is_index_extended and not is_middle_extended and not is_ring_extended:
        return "L"
    elif not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return "M"
    elif not is_index_extended and not is_middle_extended and not is_ring_extended and is_pinky_extended:
        return "N"
    elif is_thumb_folded and not is_index_extended:
        return "O"
    elif is_index_extended and is_middle_extended and not is_ring_extended and landmarks[8].x < landmarks[5].x:
        return "P"
    elif is_thumb_folded and is_index_extended and not is_middle_extended:
        return "Q"
    elif is_index_extended and is_middle_extended and not is_ring_extended and is_pinky_extended:
        return "R"
    elif not is_index_extended and is_middle_extended and not is_ring_extended:
        return "S"
    elif not is_index_extended and is_middle_extended and is_ring_extended:
        return "T"
    elif is_index_extended and is_middle_extended and is_ring_extended:
        return "U"
    elif is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
        return "V"
    elif is_index_extended and is_middle_extended and is_ring_extended and not is_pinky_extended:
        return "W"
    elif is_index_extended and landmarks[8].y > landmarks[6].y:
        return "X"
    elif is_thumb_folded and is_index_extended and landmarks[4].x < landmarks[8].x:
        return "Y"
    elif landmarks[8].y < landmarks[7].y:
        return "Z"

    return None

# Main loop for real-time ASL recognition
while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Convert the image to RGB (required by MediaPipe)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(rgb_image)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw connections between landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize the hand gesture
            gesture = classify_asl_letter(hand_landmarks, image.shape)
            if gesture:
                # Display the recognized gesture on the screen
                cv2.putText(image, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with landmarks and recognized gesture
    cv2.imshow("ASL Sign Recognition", image)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
