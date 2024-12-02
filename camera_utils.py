import cv2
import numpy as np

class CameraCapture:
    def __init__(self, model):
        self.model = model
        self.cap = None

    def preprocess_frame(self, frame):
        """Preprocess camera frame for digit recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract region of interest (ROI)
        roi = gray[100:380, 100:380]  # Fixed region for digit display
        
        # Resize to match model input
        resized = cv2.resize(roi, (28, 28))
        
        # Normalize and reshape
        normalized = resized.astype('float32') / 255.0
        processed = normalized.reshape(1, 28, 28, 1)
        
        return processed, roi

    def start_capture(self):
        """Start real-time digit recognition from camera feed"""
        self.cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw ROI rectangle
            cv2.rectangle(frame, (100, 100), (380, 380), (0, 255, 0), 2)
            
            # Process frame and make prediction
            processed_frame, roi = self.preprocess_frame(frame)
            prediction = self.model.predict(processed_frame)
            digit = np.argmax(prediction[0])
            confidence = prediction[0][digit] * 100
            
            # Display results
            cv2.putText(frame, f"Digit: {digit}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Digit Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
