from flask import Flask, Response, jsonify, render_template, request, redirect, url_for
import face_recognition
import cv2
import os
import json
from datetime import datetime
from flask_mysqldb import MySQL
import time
from scipy.spatial import distance
from collections import Counter

from flask import session

app = Flask(__name__)

app.secret_key = 'cindykangnam1!2@'

# MySQL 설정
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306  
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'cindykangnam1!2@'
app.config['MYSQL_DB'] = 'aikiosk_db'

mysql = MySQL(app)


# 전역 변수
check_start_time = None
person_detected = False
matched_once = False
matched_result = None
new_face_id = None

coffee_id = None
total_order_count = None
coffee_order_count = None
total_coffee_count = None

last_order_date = None
last_order_face_image = None

coffee_name = None
coffee_image_url = None
temp_type =None

top2_coffee_id = None
top2_coffee_name = None
top2_temp_type =None

top3_coffee_id = None
top3_coffee_name = None
top3_temp_type =None

inserted = False

top3_recent_faces_data = []

matched_unique_face_ids = []


@app.route('/get_data')
def get_data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM coffee_menu")
    data = cur.fetchall()
    cur.close()
    return jsonify(data)

@app.route('/detect_person')
def detect_person():
    global person_detected, matched_result, new_face_id , coffee_id, total_order_count , coffee_order_count, last_order_date , last_order_face_image , total_coffee_count, temp_type
    global coffee_name, coffee_image_url, top2_coffee_id, top2_coffee_name, top2_temp_type, top3_coffee_id, top3_coffee_name,top3_temp_type,top3_recent_faces_data , matched_unique_face_ids

    result = matched_result
    matched_result = None
    face_id = new_face_id

    print("Returning temp_type:", temp_type)
    print("Returning temp_type:", top2_temp_type)
    print("Returning temp_type:", top3_temp_type)

    return jsonify({
        'detected': person_detected,
        'match_result': result,
        'new_face_id': face_id,
        'coffee_id': coffee_id,
        'total_order_count': total_order_count,
        'coffee_order_count': coffee_order_count,
        'last_order_date': last_order_date,
        'last_order_face_image': last_order_face_image,
        'coffee_name': coffee_name,
        'coffee_image_url': coffee_image_url,
        'temp_type' : temp_type,
        'top2_coffee_id': top2_coffee_id,
        'top2_coffee_name': top2_coffee_name,
        'top2_temp_type' : top2_temp_type,
        'top3_coffee_id': top3_coffee_id,
        'top3_coffee_name': top3_coffee_name,
        'top3_temp_type' : top3_temp_type,
        'total_coffee_count' : total_coffee_count,
        'top3_recent_faces_data' : top3_recent_faces_data,
        'matched_unique_face_ids' : matched_unique_face_ids
    })

def gen_frames():
    global person_detected, matched_once, check_start_time, matched_result, new_face_id
    global coffee_id, total_order_count, coffee_name, coffee_image_url, inserted
    global coffee_order_count, last_order_date, last_order_face_image
    global top2_coffee_id, top2_coffee_name, top3_coffee_id, top3_coffee_name , total_coffee_count , top3_recent_faces_data

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        face_locations = face_recognition.face_locations(frame)
        person_detected = len(face_locations) > 0

        if person_detected and not matched_once:
            if check_start_time is None:
                check_start_time = time.time()

        elapsed = time.time() - check_start_time if check_start_time else 0

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if not matched_once and elapsed < 3:
                cv2.putText(frame, "Checking~", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 2)

            elif not matched_once and elapsed >= 3:
                face_image = frame[top:bottom, left:right]
                rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(rgb_face_image)

                if face_encoding and not inserted:
                    print("✅ 얼굴 인코딩 성공")
                    vector = face_encoding[0]
                    json_vector = json.dumps(vector.tolist())

                    folder_path = os.path.join("static", "faces")
                    os.makedirs(folder_path, exist_ok=True)
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"face_{timestamp_str}.jpg"
                    save_path = os.path.join(folder_path, filename)
                    cv2.imwrite(save_path, face_image)
                    print(f" 얼굴 이미지 저장됨: {save_path}")

                    try:
                        with app.app_context():
                            cur = mysql.connection.cursor()
                            cur.execute(
                                "INSERT INTO detected_faces (image_path, detected_time, face_vector) VALUES (%s, %s, %s)",
                                (save_path, now, json_vector)
                            )
                            mysql.connection.commit()
                            new_face_id = cur.lastrowid
                            print(f"✅ 얼굴 등록 완료 - ID: {new_face_id}")
                            cur.close()
                        inserted = True
                    except Exception as e:
                        print("❌ 등록 실패:", e)

                    with app.app_context():
                        cur = mysql.connection.cursor()
                        cur.execute("SELECT id, face_vector FROM detected_faces")
                        rows = cur.fetchall()

                        matched_ids = []
                        threshold = 0.6

                        for row in rows:
                            db_id = row[0]
                            db_vector_json = row[1]
                            db_vector = json.loads(db_vector_json)
                            dist = distance.euclidean(vector, db_vector)

                            if dist < threshold:
                                matched_ids.append(db_id)

                        print(f"🎯 유사도 통과한 후보들: {matched_ids}")

                        if matched_ids:
                            format_strings = ','.join(['%s'] * len(matched_ids))

                            cur.execute(f"""
                                SELECT DISTINCT face_id 
                                FROM coffee_order 
                                WHERE face_id IN ({format_strings})
                            """, tuple(matched_ids))

                            result = cur.fetchall()

                            if result:
                                matched_face_ids = [r[0] for r in result]
                                format_strings2 = ','.join(['%s'] * len(matched_face_ids))

                                global matched_unique_face_ids
                                matched_unique_face_ids = matched_face_ids

                                cur.execute(f"""
                                    SELECT c.coffee_id, cm.name, cm.image_url, c.temp_type
                                    FROM coffee_order c
                                    LEFT JOIN coffee_menu cm ON c.coffee_id = cm.id
                                    WHERE c.face_id IN ({format_strings2})
                                """, tuple(matched_face_ids))

                                orders = cur.fetchall()

                                #print("✅ Orders Raw Data:", orders)


                                if orders:
                                    coffee_counter = Counter([o[0] for o in orders])
                                    most_common = coffee_counter.most_common(3)

                                    if len(most_common) > 0:
                                        coffee_id = most_common[0][0]
                                        for o in orders:
                                            if o[0] == coffee_id:
                                                coffee_name = o[1]
                                                coffee_image_url = o[2]
                                                global temp_type
                                                temp_type = o[3]
                                                #print("Debug temp_type:", temp_type)
                                                break

                                    if len(most_common) > 1:
                                        top2_coffee_id = most_common[1][0]
                                        for o in orders:
                                            if o[0] == top2_coffee_id:
                                                top2_coffee_name = o[1]
                                                global top2_temp_type
                                                top2_temp_type = o[3]
                                                print("DEBUG top2:", top2_coffee_name, top2_temp_type)
                                                break

                                    if len(most_common) > 2:
                                        top3_coffee_id = most_common[2][0]
                                        for o in orders:
                                            if o[0] == top3_coffee_id:
                                                top3_coffee_name = o[1]
                                                global top3_temp_type
                                                top3_temp_type = o[3]
                                                print("DEBUG top3:", top3_coffee_name, top3_temp_type)
                                                break

                                    matched_result = "matched"
                                    total_order_count = len(matched_face_ids)

                                    cur.execute(f"""
                                        SELECT COUNT(DISTINCT face_id) FROM coffee_order
                                        WHERE face_id IN ({format_strings2}) AND coffee_id = %s
                                    """, tuple(matched_face_ids) + (coffee_id,))
                                    coffee_order_count = cur.fetchone()[0]

                                    #추가
                                    cur.execute(f"""
                                        SELECT COUNT(*) FROM coffee_order
                                        WHERE face_id IN ({format_strings2}) AND coffee_id = %s
                                    """, tuple(matched_face_ids) + (coffee_id,))
                                    total_coffee_count = cur.fetchone()[0]

                                    cur.execute(f"""
                                        SELECT o.face_id, o.order_time, df.image_path
                                        FROM coffee_order o
                                        JOIN detected_faces df ON o.face_id = df.id
                                        WHERE o.face_id IN ({format_strings2})
                                        ORDER BY o.order_time DESC
                                        LIMIT 1
                                    """, tuple(matched_face_ids))
                                    last_order_info = cur.fetchone()

                                    if last_order_info:
                                        last_order_date = last_order_info[1].strftime('%Y-%m-%d %H:%M:%S')
                                        last_order_face_image = last_order_info[2]
                                    else:
                                        last_order_date = None
                                        last_order_face_image = None

                                    
                                    ###########################################
                                    # 최근 방문한 top3 유사 얼굴과 커피 주문 정보 가져오기

                                    cur.execute(f"""
                                        SELECT o.face_id, df.image_path,cm.id, cm.name, cm.image_url, o.temp_type, o.order_time AS latest_order
                                        FROM coffee_order o
                                        JOIN detected_faces df ON o.face_id = df.id
                                        JOIN coffee_menu cm ON o.coffee_id = cm.id
                                        JOIN (
                                            SELECT face_id, MAX(order_time) AS max_order_time
                                            FROM coffee_order
                                            WHERE face_id IN ({format_strings2})
                                            GROUP BY face_id
                                        ) latest ON o.face_id = latest.face_id AND o.order_time = latest.max_order_time
                                        ORDER BY o.order_time DESC
                                        LIMIT 3
                                    """, tuple(matched_face_ids))

                                    top3_recent_faces = cur.fetchall()

                                    print("🧊 최근 방문한 유사 인물 top3")
                                    for i, row in enumerate(top3_recent_faces, 1):
                                        print(f"Top {i}: Face ID {row[0]}, Image: {row[1]}, Coffee: {row[2]}")
                                        # 딕셔너리로 하나씩 만들어서 리스트에 추가
                                        top3_recent_faces_data.append({
                                            'face_id': row[0],
                                            'image_path': row[1],
                                            'coffee_id': row[2],
                                            'coffee_name': row[3],
                                            'coffee_image_url': row[4],
                                            'temp_type' : row[5],
                                            'latest_order': row[6]
                                        })
                                else:
                                    matched_result = "not_matched"
                                    coffee_order_count = 0
                            else:
                                matched_result = "not_matched"
                                coffee_order_count = 0
                        else:
                            matched_result = "not_matched"
                            coffee_order_count = 0

                        matched_once = True
                        cur.close()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()




@app.route('/')
def index():
    global matched_once, matched_result, check_start_time, inserted
    global person_detected, new_face_id, coffee_id, total_order_count
    global coffee_name, coffee_image_url, coffee_order_count
    global last_order_date, last_order_face_image
    global top2_coffee_id, top2_coffee_name, top3_coffee_id, top3_coffee_name, total_coffee_count, top3_recent_faces_data , matched_unique_face_ids

    matched_once = False
    matched_result = None
    check_start_time = None
    inserted = False

    person_detected = False
    new_face_id = None
    coffee_id = None
    total_order_count = None
    coffee_order_count = None
    coffee_name = None
    coffee_image_url = None
    temp_type = None
    last_order_date = None
    last_order_face_image = None
    top2_coffee_id = None
    top2_coffee_name = None
    top2_temp_type =None

    top3_coffee_id = None
    top3_coffee_name = None
    top3_temp_type = None

    total_coffee_count = None
    top3_recent_faces_data =[]

    matched_unique_face_ids = []

    session.clear() 
    return render_template('mode.html')

@app.route('/quick_mode', methods=['GET'])
def quick_mode():
    global matched_once, matched_result, check_start_time, inserted
    global person_detected, new_face_id, coffee_id, total_order_count
    global coffee_name, coffee_image_url, coffee_order_count
    global last_order_date, last_order_face_image
    global top2_coffee_id, top2_coffee_name, top3_coffee_id, top3_coffee_name, total_coffee_count, top3_recent_faces_data , matched_unique_face_ids

    matched_once = False
    matched_result = None
    check_start_time = None
    inserted = False

    person_detected = False
    new_face_id = None
    coffee_id = None
    total_order_count = None
    coffee_order_count = None
    coffee_name = None
    coffee_image_url = None
    temp_type =None
    last_order_date = None
    last_order_face_image = None

    top2_coffee_id = None
    top2_coffee_name = None
    top2_temp_type =None

    top3_coffee_id = None
    top3_coffee_name = None
    top3_temp_name =None

    total_coffee_count = None
    top3_recent_faces_data =[]

    matched_unique_face_ids = []


    session.clear() 
    return render_template('index.html')

@app.route('/latest_mode', methods=['GET'])
def latest_mode():
    global matched_once, matched_result, check_start_time, inserted
    global person_detected, new_face_id, coffee_id, total_order_count
    global coffee_name, coffee_image_url, coffee_order_count
    global last_order_date, last_order_face_image
    global top2_coffee_id, top2_coffee_name, top3_coffee_id, top3_coffee_name, total_coffee_count, top3_recent_faces_data , matched_unique_face_ids

    matched_once = False
    matched_result = None
    check_start_time = None
    inserted = False

    person_detected = False
    new_face_id = None
    coffee_id = None
    total_order_count = None
    coffee_order_count = None
    coffee_name = None
    coffee_image_url = None
    temp_type =None
    last_order_date = None
    last_order_face_image = None
    top2_coffee_id = None
    top2_coffee_name = None
    top2_temp_type =None
    top3_coffee_id = None
    top3_coffee_name = None
    top3_temp_type =None

    total_coffee_count = None
    top3_recent_faces_data =[]

    matched_unique_face_ids = []


    session.clear() 
    return render_template('latest_index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def find_available_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 윈도우면 CAP_DSHOW 붙이는 게 안정적
        if cap.isOpened():
            print(f"✅ 카메라 열림: 장치 번호 {i}")
            available.append(i)
            cap.release()
        else:
            print(f"❌ 카메라 없음: 장치 번호 {i}")
    return available

if __name__ == '__main__':
    app.run('localhost', port=4001, debug=True)
    cameras = find_available_cameras()
    print("사용 가능한 카메라 번호:", cameras)
