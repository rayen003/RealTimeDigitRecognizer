# Real-Time Digit Recognition System

A real-time computer vision system that recognizes handwritten digits using a webcam feed. The system uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to provide instant digit recognition.

## Features

- Real-time digit recognition through webcam
- CNN model trained on MNIST dataset
- Live confidence scores for predictions
- User-friendly visualization
- Mixed precision training for better performance

## Project Structure

```
RealTimeDigitRecognizer/
├── data_loader.py      # Handles MNIST dataset operations
├── model.py            # CNN model architecture
├── camera_utils.py     # Real-time video processing
├── main.py            # Main application entry
└── requirements.txt   # Project dependencies
```

## Requirements

- Python 3.10
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[YOUR_USERNAME]/RealTimeDigitRecognizer.git
cd RealTimeDigitRecognizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. A window will open showing your webcam feed
3. Hold up a digit in the green rectangle
4. The system will display:
   - Recognized digit
   - Confidence score
5. Press 'q' to quit

## Model Architecture

The CNN model consists of:
- Input Layer (28x28x1)
- 2 Convolutional layers with MaxPooling
- Dense layer with dropout
- Output layer (10 classes)

## Performance

- Training accuracy: >98% on MNIST
- Real-time inference
- Mixed precision training enabled

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST dataset for training data
- TensorFlow team for the deep learning framework
- OpenCV team for computer vision capabilities
