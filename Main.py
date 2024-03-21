from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load pre-trained models
age_model = load_model("Age-VGG16.keras")
gender_model = load_model("Gender-ResNet152.keras")

# Function to preprocess image
def preprocess_image(image):
    # Resize image to fit model's input shape
    image = cv2.resize(image, (64, 64))
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize image
    image = image.astype('float32') / 255
    # Reshape image to match model's input shape
    image = np.reshape(image, (1, 64, 64, 1))
    return image

# Function to predict age and gender from image
def predict_age_gender(image):
    # Preprocess image
    image = preprocess_image(image)
    # Predict age
    age = age_model.predict(image)[0][0]
    # Predict gender
    gender = "Male" if gender_model.predict(image)[0][0] > 0.5 else "Female"
    return age, gender

# Function to capture video from webcam
def webcam():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Predict age and gender from frame
        age, gender = predict_age_gender(frame)
        # Draw predicted age and gender on frame
        cv2.putText(frame, f'Age: {age}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {gender}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
