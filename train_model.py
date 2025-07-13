import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

class SignLanguageTrainer:
    def __init__(self):
        self.data_dir = "sign_lang_env/sign_language_data"
        self.sequence_length = 30  # Number of frames to consider for each prediction
        self.landmark_count = 63  # 21 landmarks * 3 coordinates (x, y, z)
        self.history_file = os.path.join(self.data_dir, "training_history.json")
        # Load words from words.json and create mappings
        words_path = os.path.join(self.data_dir, "words.json")
        with open(words_path, 'r') as f:
            words_data = json.load(f)
            self.word_to_index = {word: idx for idx, word in enumerate(words_data)}
            self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
    def load_data(self):
        """Load and prepare the data for training (single-frame version)"""
        X = []
        y = []
        
        # Load samples for each word
        for word in self.word_to_index.keys():
            word_dir = os.path.join(self.data_dir, word)
            if not os.path.exists(word_dir):
                print(f"No data directory found for word: {word}")
                continue
                
            # Load all samples for this word
            samples = []
            for sample_file in os.listdir(word_dir):
                if sample_file.endswith('.npy'):
                    sample_path = os.path.join(word_dir, sample_file)
                    try:
                        sample = np.load(sample_path)
                        if sample.shape == (self.landmark_count,):
                            samples.append(sample)
                        else:
                            print(f"Sample {sample_file} has unexpected shape {sample.shape}, skipping.")
                    except Exception as e:
                        print(f"Error loading sample {sample_file}: {e}")
                        continue
            if samples:
                print(f"Loading {len(samples)} samples for word: {word}")
                X.extend(samples)
                # Create one-hot encoded labels
                label = np.zeros(len(self.word_to_index))
                label[self.word_to_index[word]] = 1
                y.extend([label] * len(samples))
        if not X:
            raise ValueError("No valid samples found for training")
        return np.array(X), np.array(y)
    
    def create_model(self):
        """Create and return a model for single-frame input"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.landmark_count,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.word_to_index), activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def load_or_create_model(self, num_classes):
        """Load existing model or create a new one"""
        model_path = 'sign_language_model.h5'
        if os.path.exists(model_path):
            print("Loading existing model...")
            return load_model(model_path)
        else:
            print("Creating new model...")
            return self.create_model()
    
    def save_training_history(self, history):
        """Save training history"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                existing_history = json.load(f)
        else:
            existing_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        # Append new history
        for key in existing_history.keys():
            if key in history:
                existing_history[key].extend(history[key])
        
        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(existing_history, f)
    
    def plot_training_history(self):
        """Plot training history"""
        if not os.path.exists(self.history_file):
            print("No training history found.")
            return
            
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def train(self):
        """Train the model"""
        # Load and prepare the data
        X, y = self.load_data()
        
        # Create and compile the model
        model = self.create_model()
        
        # Add data augmentation
        def augment_data(X, y):
            augmented_X = []
            augmented_y = []
            
            for i in range(len(X)):
                # Original data
                augmented_X.append(X[i])
                augmented_y.append(y[i])
                
                # Add noise
                noise = np.random.normal(0, 0.01, X[i].shape)
                augmented_X.append(X[i] + noise)
                augmented_y.append(y[i])
                
                # Scale
                scale = np.random.uniform(0.9, 1.1)
                augmented_X.append(X[i] * scale)
                augmented_y.append(y[i])
            
            return np.array(augmented_X), np.array(augmented_y)
        
        # Augment the data
        X_aug, y_aug = augment_data(X, y)
        
        # Print shapes for debugging
        print(f"X_aug shape before reshape: {X_aug.shape}")
        print(f"y_aug shape: {y_aug.shape}")
        # No reshape needed for single-frame input
        
        # Train the model with better parameters
        history = model.fit(
            X_aug, y_aug,
            epochs=500,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
        )
        
        # Save the model
        model.save('sign_language_model.h5')
        print("Best model saved as 'best_model.h5'")
        
        # Save training history
        self.save_training_history(history.history)
        
        # Plot training history
        self.plot_training_history()
        
        return model

if __name__ == "__main__":
    trainer = SignLanguageTrainer()
    history = trainer.train()
    print("\nTraining completed! Model saved as 'sign_language_model.h5'")
    print("Training history saved and plotted in 'training_history.png'") 