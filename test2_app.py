from flask import Flask, Response, jsonify , current_app
import face_recognition
import cv2
import os
import json
from datetime import datetime
import requests
from flask_mysqldb import MySQL
import time

app = Flask(__name__)

# MySQL 설정
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306  
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'cindykangnam1!2@'
app.config['MYSQL_DB'] = 'aikiosk_db'

mysql = MySQL(app)

# 전역 변수
person_detected = False
photo_taken = False  # 사진을 이미 찍었는지 여부

@app.route('/get_data')
def get_data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM coffee_menu")
    data = cur.fetchall()
    cur.close()
    return jsonify(data)


def gen_frames():
    global person_detected, photo_taken

    cap = cv2.VideoCapture(0)
    cheese_start_time = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB 변환은 한번만
        face_locations = face_recognition.face_locations(rgb_frame)
        person_detected = len(face_locations) > 0

        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if not photo_taken:
                if cheese_start_time is None:
                    cheese_start_time = time.time()

                elapsed = time.time() - cheese_start_time
                cv2.putText(frame, "Cheese~ ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 255), 3)

                if elapsed >= 3:
                    photo_taken = True
                    cheese_start_time = None

                    # 얼굴 이미지 저장
                    face_image = frame[top:bottom, left:right]
                    folder_path = os.path.join("static", "faces")
                    os.makedirs(folder_path, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"face_{timestamp}.jpg"
                    save_path = os.path.join(folder_path, filename)
                    cv2.imwrite(save_path, face_image)
                    print(f" 얼굴 이미지 저장됨: {save_path}")

                    try:
                        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])
                        if face_encoding:
                            print("✅ 얼굴 인코딩 성공")
                            vector = face_encoding[0].tolist()
                            json_vector = json.dumps(vector)

                            # ✅ 여기서 app context 열어줍니다
                            with app.app_context():
                                cur = mysql.connection.cursor()
                                sql = "INSERT INTO detected_faces (image_path, detected_time, face_vector) VALUES (%s, %s, %s)"
                                print("📦 SQL:", sql)
                                print("📦 파라미터:", (save_path, datetime.now(), json_vector))
                                cur.execute(sql, (save_path, datetime.now(), json_vector))
                                mysql.connection.commit()
                                cur.close()
                        else:
                            print("❌ 얼굴 인코딩 실패: 인식된 얼굴 없음")

                    except Exception as e:
                        print("❗ DB 저장 또는 인코딩 중 오류 발생:", e)

            else:
                cheese_start_time = None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return "<h2>실시간 얼굴 인식 테스트</h2><img src='/video'>"

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_person')
def detect_person():
    global person_detected
    return jsonify({'detected': person_detected})

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)
