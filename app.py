from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import numpy as np
from datetime import datetime
import base64

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

known_faces = []
face_id_map = {}
next_id = 1

def get_face_encodings(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_encodings = []
    for face in faces:
        shape = shape_predictor(gray, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
        face_encodings.append((face, face_encoding))
    return face_encodings

def generate_frames():
    global next_id, face_id_map, known_faces

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face_encodings = get_face_encodings(frame)
            new_ids = set()
            for face, face_encoding in face_encodings:
                matches = dlib.chinese_whispers_clustering([np.array(face_encoding)] + known_faces, 0.6)
                id = None
                for match in matches:
                    if match != -1:
                        id = match
                        break
                if id is None:
                    id = next_id
                    known_faces.append(face_encoding)
                    next_id += 1
                new_ids.add(id)
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            for face_id in face_id_map.keys() - new_ids:
                face_id_map.pop(face_id, None)
            for face_id in new_ids - face_id_map.keys():
                face_id_map[face_id] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
def log():
    return jsonify(face_id_map)

if __name__ == "__main__":
    app.run(host="0.0.0.0, port=5000")
