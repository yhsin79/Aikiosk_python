import face_recognition
import cv2
import json
import numpy as np
from flask import Flask, Response, jsonify
from flask_mysqldb import MySQL
from datetime import datetime

app = Flask(__name__)

# MySQL 설정
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'cindykangnam1!2@'
app.config['MYSQL_DB'] = 'aikiosk_db'

mysql = MySQL(app)

@app.route('/find_match')
def find_match():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': '카메라 캡처 실패'})

    # 얼굴 인식
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return jsonify({'error': '얼굴 인식 실패'})

    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

    with app.app_context():
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, face_vector, image_path FROM detected_faces")
        results = cur.fetchall()
        cur.close()

    known_encodings = []
    ids = []
    paths = []

    for row in results:
        ids.append(row[0])
        paths.append(row[2])
        vector = json.loads(row[1])
        known_encodings.append(np.array(vector))

    # 얼굴 비교
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)

    matched_id = ids[best_match_index]
    matched_path = paths[best_match_index]
    matched_distance = distances[best_match_index]

    return jsonify({
        'matched_id': matched_id,
        'matched_path': matched_path,
        'similarity_score': float(1 - matched_distance)  # 1에 가까울수록 유사
    })
