import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

class SignLanguageInterpreter:
    def __init__(self):
        """Initialize the sign language interpreter"""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model('best_model.h5')
            print("Model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Load word mappings
        try:
            with open('words.json', 'r') as f:
                words_data = json.load(f)
                self.word_to_index = {word: idx for idx, word in enumerate(words_data)}
                self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
            print("Word mappings loaded successfully!")
            print(f"Available words: {list(self.word_to_index.keys())}")
        except Exception as e:
            print(f"Error loading word mappings: {e}")
            self.word_to_index = {}
            self.index_to_word = {}
        
        # Initialize sequence for predictions
        self.sentence = []
        self.threshold = 0.1
        self.clear_message = ""
        self.clear_message_timer = 0
        self.last_prediction = None
        self.prediction_cooldown = 0
        self.frame_count = 0
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        
        # Set up video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture device")
            raise RuntimeError("Could not open video capture device")
        print("Video capture initialized successfully!")
        
        # After loading model and word mappings, check output shape
        if self.model is not None and hasattr(self, 'word_to_index'):
            model_classes = self.model.output_shape[-1]
            word_classes = len(self.word_to_index)
            if model_classes != word_classes:
                print(f"WARNING: Model output classes ({model_classes}) do not match number of words ({word_classes})!")
        
        # Initialize training data storage
        self.training_data_dir = "training_data"
        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)
        
        # Load existing training data
        self.all_training_data = []
        self.all_training_labels = []
        self.load_existing_training_data()
        
        # Initialize live training data
        self.live_training_data = []
        self.live_training_labels = []
        self.live_training_counter = 0
        self.training_mode = False
        self.selected_word = None
        self.scroll_offset = 0  # For scrolling the word list in training mode
        self.max_visible_words = 10  # Number of words to show at once in the panel
        
    def load_existing_training_data(self):
        """Load all existing training data from files"""
        try:
            for word in self.word_to_index.keys():
                data_file = os.path.join(self.training_data_dir, f"{word}_data.npy")
                if os.path.exists(data_file):
                    data = np.load(data_file)
                    labels = np.full(len(data), self.word_to_index[word])
                    self.all_training_data.extend(data)
                    self.all_training_labels.extend(labels)
            print(f"Loaded {len(self.all_training_data)} existing training samples")
        except Exception as e:
            print(f"Error loading existing training data: {e}")

    def save_training_data(self, word):
        """Save training data for a specific word"""
        try:
            data_file = os.path.join(self.training_data_dir, f"{word}_data.npy")
            if os.path.exists(data_file):
                existing_data = np.load(data_file)
                new_data = np.array(self.live_training_data)
                combined_data = np.vstack([existing_data, new_data])
                np.save(data_file, combined_data)
            else:
                np.save(data_file, np.array(self.live_training_data))
            print(f"Saved {len(self.live_training_data)} new samples for {word}")
        except Exception as e:
            print(f"Error saving training data: {e}")

    def clear_buffers(self):
        """Clear both sentence and sequence buffers"""
        self.sentence = []
        self.last_prediction = None
        self.clear_message = "Buffers cleared!"
        self.clear_message_timer = 30
        self.frame_count = 0
        print("All buffers cleared!")

    def enter_training_mode(self):
        """Enter training mode and display available words"""
        self.training_mode = True
        self.selected_word = None
        self.scroll_offset = 0  # Always reset scroll when entering training mode
        print("\n=== Training Mode ===")
        print("Available words:")
        for i, word in enumerate(self.word_to_index.keys(), 1):
            print(f"{i}. {word}")
        print("\nPress the number key (1-10) to select a word for training")
        print("Press 't' to start training with the selected word")
        print("Press 'q' to exit training mode")
        
    def trigger_live_training(self):
        """Trigger live training with the selected word"""
        if not self.selected_word:
            self.clear_message = "Please select a word first!"
            self.clear_message_timer = 30
            return
            
        if len(self.live_training_data) < 30:
            self.clear_message = f"Need more samples! Current: {len(self.live_training_data)}/30"
            self.clear_message_timer = 30
            return
            
        print(f"\nStarting live training for word: {self.selected_word}")
        print(f"Training samples: {len(self.live_training_data)}")
        
        # Save the new training data
        self.save_training_data(self.selected_word)
        
        # Add new data to all training data
        self.all_training_data.extend(self.live_training_data)
        self.all_training_labels.extend([self.word_to_index[self.selected_word]] * len(self.live_training_data))
        
        # Convert all data to numpy arrays
        X = np.array(self.all_training_data)
        y = np.array(self.all_training_labels)
        
        # One-hot encode labels
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.word_to_index))
        
        # Create a new model if none exists
        if self.model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(self.word_to_index), activation='softmax')
            ])
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Train the model on all data
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the model
        self.model.save('best_model.h5')
        
        # Clear live training data
        self.live_training_data = []
        self.live_training_labels = []
        self.live_training_counter = 0
        
        self.clear_message = "Training completed! Model updated."
        self.clear_message_timer = 30
        self.training_mode = False
        self.selected_word = None

    def process_frame(self, frame):
        """Process a single frame and return the processed frame"""
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
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
                
                # Make prediction if we have a model
                if self.model is not None:
                    # Reshape landmarks for prediction
                    landmarks = np.array(landmarks).reshape(1, -1)
                    
                    # Get prediction
                    prediction = self.model.predict(landmarks, verbose=0)[0]
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[predicted_class]
                    
                    # Get the predicted word
                    predicted_word = self.index_to_word.get(predicted_class, "Unknown")
                    
                    # Update prediction display
                    if confidence > self.threshold:
                        # Only speak if the word has changed
                        if predicted_word != self.last_prediction:
                            self.last_prediction = predicted_word
                            self.speak_text(predicted_word)
                    
                    # Display prediction
                    cv2.putText(
                        frame,
                        f"Prediction: {predicted_word} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # If in training mode, collect data
                    if self.training_mode and self.selected_word:
                        if self.frame_count % 5 == 0:  # Collect every 5th frame
                            self.live_training_data.append(landmarks[0])
                            self.live_training_labels.append(self.word_to_index[self.selected_word])
                            self.live_training_counter += 1
                            
                            # Display training progress
                            cv2.putText(
                                frame,
                                f"Training {self.selected_word}: {self.live_training_counter} samples",
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2
                            )
        
        # Display current mode
        mode_text = "Training Mode" if self.training_mode else "Interpretation Mode"
        cv2.putText(
            frame,
            f"Mode: {mode_text}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Display clear message if any
        if self.clear_message_timer > 0:
            cv2.putText(
                frame,
                self.clear_message,
                (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            self.clear_message_timer -= 1
        
        return frame

    def speak_text(self, text):
        """Speak the given text"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error speaking text: {e}")

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, r, line_type=cv2.LINE_AA):
        # Draw a rounded rectangle by combining rectangles and ellipses
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, line_type)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, line_type)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, line_type)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, line_type)
        else:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, line_type)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, line_type)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, line_type)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, line_type)

    def draw_text_with_shadow(self, img, text, org, font, font_scale, color, thickness, shadow_color=(0,0,0), shadow_offset=(2,2)):
        x, y = org
        cv2.putText(img, text, (x + shadow_offset[0], y + shadow_offset[1]), font, font_scale, shadow_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_help_overlay(self, frame):
        overlay = frame.copy()
        alpha = 0.7
        box_width, box_height = 370, 200
        r = 20
        # Gradient background
        for i in range(box_height):
            color = (30 + i//8, 30 + i//8, 60 + i//4)
            cv2.line(overlay, (5, 5 + i), (5 + box_width, 5 + i), color, 1)
        # Rounded rectangle border
        self.draw_rounded_rectangle(overlay, (5, 5), (5 + box_width, 5 + box_height), (80, 80, 200), 2, r)
        help_lines = [
            "Controls:",
            "q: Quit",
            "c: Clear buffers",
            "m: Enter training mode",
            "1-9, 0: Select word (training mode)",
            "t: Train selected word (training mode)",
        ]
        y0 = 35
        for i, line in enumerate(help_lines):
            y = y0 + i * 28
            if i == 0:
                self.draw_text_with_shadow(overlay, line, (25, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
            else:
                self.draw_text_with_shadow(overlay, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_training_mode_overlay(self, frame):
        overlay = frame.copy()
        alpha = 0.8
        words = list(self.word_to_index.keys())
        n_words = len(words)
        box_width, box_height = 420, 60 + 36 * n_words
        r = 20
        x0, y0 = 360, 5
        # Gradient background
        for i in range(box_height):
            color = (0 + i//8, 30 + i//6, 80 + i//3)
            cv2.line(overlay, (x0, y0 + i), (x0 + box_width, y0 + i), color, 1)
        # Rounded rectangle border
        self.draw_rounded_rectangle(overlay, (x0, y0), (x0 + box_width, y0 + box_height), (0, 255, 255), 2, r)
        self.draw_text_with_shadow(overlay, "Training Mode: Select a word", (x0 + 20, y0 + 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2)
        for i, word in enumerate(words):
            y = y0 + 70 + i * 36
            num = i + 1 if i < 9 else 0
            text = f"{num}: {word}"
            color = (0, 255, 0) if self.selected_word == word else (255, 255, 255)
            self.draw_text_with_shadow(overlay, text, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_landscape_gui(self, frame):
        cam_h, cam_w = frame.shape[:2]
        panel_w = 420  # Increased width for more space
        total_w = cam_w + panel_w
        canvas = np.zeros((cam_h, total_w, 3), dtype=np.uint8)
        canvas[:, :panel_w] = (60, 60, 60)
        canvas[:, panel_w:panel_w+cam_w] = frame
        x0 = 20
        y = 40
        line_gap = 22  # Slightly smaller gap for more lines
        font_scale = 0.48  # Smaller font for more words
        # Mode
        mode_text = "Training Mode" if self.training_mode else "Interpretation Mode"
        color = (0, 255, 255) if self.training_mode else (200, 200, 255)
        cv2.putText(canvas, f"Mode: {mode_text}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += line_gap * 2
        # Help
        help_lines = [
            "Controls:",
            "q: Quit",
            "c: Clear buffers",
            "m: Training mode",
            "1-9,0: Select word (training)",
            "t: Train selected word",
            "Up/Down: Scroll words",
        ]
        for line in help_lines:
            cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1)
            y += line_gap
        y += line_gap // 2
        if self.training_mode:
            cv2.putText(canvas, "Words:", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y += line_gap
            words = list(self.word_to_index.keys())
            n_words = len(words)
            max_scroll = max(0, n_words - self.max_visible_words)
            self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
            start = self.scroll_offset
            end = min(start + self.max_visible_words, n_words)
            # Indicate if there are more words above
            if start > 0:
                cv2.putText(canvas, "▲", (x0 + 180, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
            for i in range(start, end):
                word = words[i]
                num = i + 1 if i < 9 else 0
                color = (0, 255, 0) if self.selected_word == word else (220, 220, 220)
                # Word wrapping if too long
                word_text = f"{num}: {word}"
                (text_w, _), _ = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                if text_w > (panel_w - 2 * x0):
                    # Split word if too long
                    max_chars = int((panel_w - 2 * x0) / (text_w / len(word_text)))
                    first_line = word_text[:max_chars]
                    second_line = word_text[max_chars:]
                    cv2.putText(canvas, first_line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                    y += line_gap
                    if second_line:
                        cv2.putText(canvas, second_line, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                        y += line_gap
                else:
                    cv2.putText(canvas, word_text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                    y += line_gap
            # Indicate if there are more words below
            if end < n_words:
                cv2.putText(canvas, "▼", (x0 + 180, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
            y += line_gap // 2
        if self.clear_message_timer > 0:
            cv2.putText(canvas, self.clear_message, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        return canvas

    def run(self):
        """Run the sign language interpreter"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Compose landscape GUI
                gui_frame = self.draw_landscape_gui(processed_frame)
                
                # Display the frame
                cv2.imshow('Sign Language Interpreter', gui_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if self.training_mode:
                    if key == ord('q'):
                        self.training_mode = False
                        self.selected_word = None
                        self.clear_message = "Exited training mode."
                        self.clear_message_timer = 30
                    elif key == 82:  # Up arrow
                        if self.scroll_offset > 0:
                            self.scroll_offset -= 1
                    elif key == 84:  # Down arrow
                        words = list(self.word_to_index.keys())
                        if self.scroll_offset + self.max_visible_words < len(words):
                            self.scroll_offset += 1
                    elif ord('1') <= key <= ord('9'):
                        word_index = key - ord('1') + self.scroll_offset
                        words = list(self.word_to_index.keys())
                        if word_index < len(words):
                            self.selected_word = words[word_index]
                            self.clear_message = f"Selected word: {self.selected_word}"
                            self.clear_message_timer = 30
                            print(f"\nSelected word for training: {self.selected_word}")
                            print("Press 't' to start training when ready")
                    elif key == ord('0'):
                        words = list(self.word_to_index.keys())
                        idx = 9 + self.scroll_offset
                        if len(words) > idx:
                            self.selected_word = words[idx]
                            self.clear_message = f"Selected word: {self.selected_word}"
                            self.clear_message_timer = 30
                            print(f"\nSelected word for training: {self.selected_word}")
                            print("Press 't' to start training when ready")
                    elif key == ord('t'):
                        if self.selected_word:
                            self.trigger_live_training()
                        else:
                            self.clear_message = "Please select a word first!"
                            self.clear_message_timer = 30
                    elif key == ord('c'):
                        self.clear_buffers()
                else:
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        self.clear_buffers()
                    elif key == ord('m'):
                        self.scroll_offset = 0
                        self.enter_training_mode()
                
                self.frame_count += 1
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    interpreter.run() 