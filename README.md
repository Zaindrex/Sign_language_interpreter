# Sign Language Interpreter

A real-time sign language interpreter using computer vision and deep learning. This project allows users to interpret hand signs via webcam and supports live training to add new signs or improve recognition accuracy.

## Features

- **Real-time sign language recognition** using webcam and MediaPipe.
- **Live training mode:** Add new samples for existing words or new words on the fly.
- **Customizable vocabulary:** Easily expand the set of recognized words.
- **Text-to-speech output** for recognized signs.
- **User-friendly GUI** with clear instructions and feedback.

---

## Project Structure

```
.
├── sign_language_interpreter.py   # Main interpreter (run this to start)
├── collect_data.py               # Tool for collecting new sign data
├── train_model.py                # Script to train or retrain the model
├── best_model.h5                 # Pre-trained model (updated after training)
├── requirements.txt              # Python dependencies
├── training_data/                # Stores .npy files for each word's samples
├── words.json                    # List of recognized words
├── index_to_word.npy, word_to_index.npy # Word-index mappings
├── training_history.png/json     # Training progress and metrics
└── README.md                     # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "final year project"
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv sign_lang_env
# On Windows:
sign_lang_env\Scripts\activate
# On Mac/Linux:
source sign_lang_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Interpreter

```bash
python sign_language_interpreter.py
```

- The webcam window will open.
- Recognized signs will be displayed and spoken aloud.
- **Controls:**
  - `q`: Quit
  - `c`: Clear buffers
  - `m`: Enter training mode
  - `1-9, 0`: Select word (in training mode)
  - `t`: Train selected word (in training mode)
  - Arrow keys: Scroll word list (in training mode)

### 2. Live Training (Add More Samples)

- Press `m` to enter training mode.
- Use number keys to select a word.
- Press `t` to start collecting samples for the selected word.
- After collecting enough samples, the model will retrain and update automatically.

### 3. Add New Words or Collect Data

To add a new word or collect more data for existing words:

```bash
python collect_data.py
```

- Follow the prompts to add a new word or collect samples.
- Data will be saved in `training_data/`.

### 4. Train or Retrain the Model

After collecting new data, retrain the model:

```bash
python train_model.py
```

- The script will load all data, train the model, and save the best model as `best_model.h5`.
- Training history and metrics will be saved and plotted.

---

## Data Files

- **training_data/**: Contains `.npy` files for each word (e.g., `hello_data.npy`).
- **words.json**: List of recognized words.
- **index_to_word.npy, word_to_index.npy**: Mappings for model output.
- **best_model.h5**: The trained model file.

---

## Requirements

See `requirements.txt` for all dependencies. Key packages:

- opencv-python
- mediapipe
- numpy
- tensorflow
- pyttsx3
- matplotlib
- scikit-learn

---

## Notes

- Ensure your webcam is connected and not used by another application.
- You can expand the vocabulary by adding new words and collecting samples.
- The model supports live retraining for improved accuracy.

---

## Credits

- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking.
- [TensorFlow](https://www.tensorflow.org/) for deep learning.
- [OpenCV](https://opencv.org/) for computer vision.

---

## License

MIT License (or specify your license here)
