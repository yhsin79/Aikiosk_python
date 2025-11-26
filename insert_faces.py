from flask import Flask, jsonify
import face_recognition
import os
import json
from datetime import datetime
from flask_mysqldb import MySQL

app = Flask(__name__)

# MySQL 설정
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306  
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'cindykangnam1!2@'
app.config['MYSQL_DB'] = 'aikiosk_db'

mysql = MySQL(app)

@app.route('/')
def insert_face_vectors():
    folder_path = r'C:\Users\HYESHIN YU\Downloads\LFW-emotion-dataset\LFW-emotion-dataset\data\LFW-FER\LFW-FER\train\positive'
    
    if not os.path.exists(folder_path):
        return f"❌ 폴더 경로가 존재하지 않습니다: {folder_path}", 400

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(files)
    success_count = 0

    print(f"📂 총 {total}개의 이미지 처리 시작")

    for i, filename in enumerate(files):
        full_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(full_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            vector = face_encodings[0]
            json_vector = json.dumps(vector.tolist())

            try:
                cur = mysql.connection.cursor()
                sql = "INSERT INTO detected_faces (image_path, detected_time, face_vector) VALUES (%s, %s, %s)"
                cur.execute(sql, ("", datetime.now(), json_vector))
                mysql.connection.commit()
                cur.close()
                success_count += 1
                print(f"✅ [{i+1}/{total}] 얼굴 벡터 저장 성공: {filename}")
            except Exception as e:
                print(f"❗ DB 저장 오류 ({filename}):", e)
        else:
            print(f"⚠️  [{i+1}/{total}] 얼굴 인식 실패: {filename}")

    return jsonify({
        "total": total,
        "success": success_count,
        "failed": total - success_count
    })

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
