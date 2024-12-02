import tensorflow as tf
from data_loader import DigitDataLoader
from model import DigitRecognitionModel
from camera_utils import CameraCapture

def main():
    print("Starting Digit Recognition System...")
    
    # Enable mixed precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Initialize components
    data_loader = DigitDataLoader()
    model = DigitRecognitionModel()
    
    # Load and prepare data
    print("Loading digit dataset...")
    data_loader.download_dataset()
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    # Train model
    print("Training digit recognition model...")
    model.train(X_train, y_train, X_test, y_test, epochs=5)
    
    # Start real-time recognition
    print("Starting camera for real-time digit recognition...")
    camera = CameraCapture(model)
    camera.start_capture()

if __name__ == "__main__":
    main()
