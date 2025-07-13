import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json

class SignLanguageDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create directories for data collection
        self.data_dir = "sign_lang_env/sign_language_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Load existing words or initialize with default words
        self.words_file = os.path.join(self.data_dir, "words.json")
        if os.path.exists(self.words_file):
            with open(self.words_file, 'r') as f:
                self.words = json.load(f)
        else:
            self.words = [
                "hello",
                "thank you",
                "please",
                "sorry",
                "goodbye",
                "yes",
                "no",
                "help",
                "water",
                "food"
            ]
            # Save initial words
            with open(self.words_file, 'w') as f:
                json.dump(self.words, f)
    
    def check_camera(self):
        """Check if camera is available and working"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera. Please check if your camera is connected and not in use by another application.")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("Error: Could not read frame from camera.")
            return False
            
        return True
        
    def add_new_word(self, word):
        """Add a new word to the vocabulary"""
        if word not in self.words:
            self.words.append(word)
            # Save updated words list
            with open(self.words_file, 'w') as f:
                json.dump(self.words, f)
            print(f"Added new word: {word}")
        else:
            print(f"Word '{word}' already exists in vocabulary")
    
    def get_next_sample_number(self, word_dir):
        """Get the next sample number for a word"""
        if not os.path.exists(word_dir):
            return 0
        existing_samples = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
        if not existing_samples:
            return 0
        # Get the highest sample number and add 1
        sample_numbers = [int(s.split('_')[1].split('.')[0]) for s in existing_samples]
        return max(sample_numbers) + 1
        
    def collect_data(self, word, num_samples=100):
        """Collect data for a specific word"""
        if not self.check_camera():
            return
            
        word_dir = os.path.join(self.data_dir, word)
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)
            
        # Get the next sample number to continue from existing data
        start_sample = self.get_next_sample_number(word_dir)
        total_samples = start_sample + num_samples
            
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\nCollecting data for word: {word}")
        print(f"Starting from sample {start_sample}")
        print("Press 'c' to capture a frame, 'q' to quit")
        
        sample_count = start_sample
        last_capture_time = 0
        capture_cooldown = 0.5  # Minimum time between captures in seconds
        
        try:
            while sample_count < total_samples:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                    
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and detect hands
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Extract landmarks
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Save landmarks when 'c' is pressed
                        current_time = time.time()
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('c') and (current_time - last_capture_time) > capture_cooldown:
                            # Save landmarks to file
                            landmark_file = os.path.join(word_dir, f"sample_{sample_count}.npy")
                            np.save(landmark_file, landmarks)
                            sample_count += 1
                            last_capture_time = current_time
                            print(f"Collected sample {sample_count}/{total_samples}")
                            
                            # Visual feedback
                            cv2.putText(
                                frame,
                                "CAPTURED!",
                                (frame.shape[1]//2 - 100, frame.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 255, 0),
                                3
                            )
                            cv2.imshow('Sign Language Data Collection', frame)
                            cv2.waitKey(500)  # Show feedback for 500ms
                            
                        elif key == ord('q'):
                            print("Quitting data collection...")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                
                # Display the frame
                cv2.putText(
                    frame,
                    f"Word: {word} - Samples: {sample_count}/{total_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Add instructions
                cv2.putText(
                    frame,
                    "Press 'c' to capture, 'q' to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Sign Language Data Collection', frame)
                
                # Limit frame rate
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
    def collect_all_words(self, samples_per_word=100):
        """Collect data for all words"""
        for word in self.words:
            self.collect_data(word, samples_per_word)
            print(f"\nCompleted data collection for: {word}")
            time.sleep(2)  # Pause between words
            
        print("\nData collection completed for all words!")

if __name__ == "__main__":
    collector = SignLanguageDataCollector()
    collector.collect_all_words(samples_per_word=100) 