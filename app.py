from flask import Flask, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS   # 🆕 Add this line
import cv2
import numpy as np
import dlib
import os
import json

app = Flask(__name__)
CORS(app)  # 🆕 Enable CORS for all routes
application = app

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmark_68_model.dat")

# Landmark indices for the mouth
MOUTH = list(range(48, 68))

def calculate_smile_score(landmarks, face_height):
    left_corner = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right_corner = np.array([landmarks.part(54).x, landmarks.part(54).y])
    upper_lip = np.array([landmarks.part(51).x, landmarks.part(51).y])
    lower_lip = np.array([landmarks.part(57).x, landmarks.part(57).y])

    mouth_width = np.linalg.norm(left_corner - right_corner)
    mouth_height = np.linalg.norm(upper_lip - lower_lip)
    lip_curvature = np.linalg.norm(
        np.array([landmarks.part(50).x, landmarks.part(50).y]) -
        np.array([landmarks.part(58).x, landmarks.part(58).y])
    )

    mouth_openness = (mouth_height / face_height) * 100
    curvature_effect = (lip_curvature / face_height) * 80
    smile_score = mouth_openness + curvature_effect
    smile_score = max(0, min(smile_score, 100))

    if smile_score > 10:
        smile_score = ((smile_score - 10) / (50 - 10)) * 90 + 10
        smile_score = min(100, smile_score)

    return round(smile_score, 2)

def draw_landmarks(image, landmarks, color=(255, 0, 0)):
    for i in MOUTH:
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 2, color, -1)

def draw_mouth_curve(image, landmarks, color=(0, 255, 255)):
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH], np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = "uploaded_image.jpg"
    file.save(image_path)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    smile_scores = []

    for face in faces:
        landmarks = predictor(gray, face)
        face_height = face.bottom() - face.top()
        smile_score = calculate_smile_score(landmarks, face_height)
        smile_scores.append(smile_score)

        draw_landmarks(image, landmarks)
        draw_mouth_curve(image, landmarks)

        color = (0, 255, 0) if smile_score > 50 else (0, 0, 255)
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        cv2.putText(image, f"Smile: {smile_score}%", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    processed_image_path = "output.jpg"
    cv2.imwrite(processed_image_path, image)

    response = make_response(send_file(processed_image_path, mimetype='image/jpeg'))
    response.headers['smile_scores'] = json.dumps(smile_scores)
    return response

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))

