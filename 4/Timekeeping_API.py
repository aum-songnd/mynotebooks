# Flask and related extensions
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy

# Authentication and encryption
import bcrypt

# Image processing and computer vision
import cv2
import numpy as np
from PIL import Image

# General utilities
import os
import uuid
import math
import logging
from io import BytesIO
from datetime import datetime, date, timedelta

# Machine learning and face recognition
import tensorflow as tf
import face_recognition
from keras_facenet import FaceNet

# Database and serialization
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle

from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

MyFaceNet = FaceNet()

app = Flask(__name__)
socketio = SocketIO(app)

# Khóa bí mật cho JWT (cần được bảo mật và không chia sẻ)
SECRET_KEY = 'gMfmHDc6QiNTl8zA'
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JWT_SECRET_KEY'] = SECRET_KEY
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# Configure database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123456789@localhost:5432/Timekeeper'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Khởi tạo tiện ích mở rộng
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Tải mô hình chống giả mạo khuôn mặt
model_path = "model/output_model.h5"
le_path = "model/output_label_encoder.pickle"
liveness_model = tf.keras.models.load_model(model_path)
le = pickle.load(open(le_path, 'rb'))

# Tạo một ThreadPoolExecutor để xử lý song song
executor = ThreadPoolExecutor(max_workers=4)  # Số luồng tối đa có thể xử lý cùng lúc

# Define models
class Account(db.Model):
    username = db.Column(db.String(50), primary_key=True)
    password = db.Column(db.String(128), nullable=False)
    hint = db.Column(db.Text)
    company = db.Column(db.String(20), unique=True)  # Add unique constraint

class Employee(db.Model):   
    employeeID = db.Column(db.String(10), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    birthday = db.Column(db.Date)
    gender = db.Column(db.String(10))
    address = db.Column(db.Text)
    phone = db.Column(db.String(15))
    company = db.Column(db.String(20), db.ForeignKey('account.company', ondelete='CASCADE'), nullable=False)

class Timekeeping(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    employeeID = db.Column(db.String(10), db.ForeignKey('employee.employeeID', ondelete='CASCADE'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(50))


#######################################################################################
# Route để đăng ký người dùng mới
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    hint = data.get('hint')
    company = data.get('company')

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    new_user = Account(username=username, password=hashed_password, hint=hint, company=company)
    
    try:
        db.session.add(new_user)
        db.session.commit()

        client = QdrantClient(url="http://localhost:6333")

        # Tạo bộ sưu tập nhân viên
        client.recreate_collection(
            collection_name=company,
            vectors_config=VectorParams(size=512, distance=Distance.EUCLID),
        )

    except Exception as e:
        db.session.rollback()
        if "unique constraint" in str(e):
            return jsonify({"message": "User already exists"}), 409
        else:
            return jsonify({"message": str(e)}), 500

    return jsonify({"message": "User registered successfully"}), 200


#######################################################################################
# Route để đăng nhập và tạo token JWT
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    print(f"username: {username}")
    print(f"password: {password}")

    try:
        user = Account.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            token = create_access_token(identity={'username': username, 'role': 'admin', 'company': user.company})
            return jsonify({"token": token, 'company': user.company}), 200
        else:
            return jsonify({"message": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# Route để truy cập tài nguyên bảo vệ bởi JWT
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    identity = get_jwt_identity()
    return jsonify({"message": f"Welcome {identity['username']}!"}), 200

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    distance = 1 - similarity  # Convert similarity to distance
    
    return distance


#######################################################################################
def process_face_for_location(face_location, rgb_small_frame, collection):
    # Xử lý từng khuôn mặt độc lập
    (top, right, bottom, left) = face_location
    face = rgb_small_frame[top:bottom, left:right]
    face = Image.fromarray(face)
    face_preds = face.resize((64, 64))
    face_preds = np.asarray(face_preds)
    face_preds = np.expand_dims(face_preds, axis=0)

    preds = liveness_model.predict(face_preds)[0]
    j = np.argmax(preds)
    label = le.classes_[j]

    result = {"id": 'Unknown', "label": label}

    if label == 'real':
        face = face.resize((160, 160))
        face = np.asarray(face)
        face = np.expand_dims(face, axis=0)

        embedding = MyFaceNet.embeddings(face)
        signature = np.squeeze(embedding, axis=0)

        client = QdrantClient(url="http://localhost:6333")
        search_result = client.search(
            collection_name=collection, query_vector=signature, with_vectors=True, limit=1
        )

        threshold = 0.35

        for scored_point in search_result:
            # vector_db = scored_point.vector
            # euclidean_score = scored_point.score
            # cosine_score = cosine_similarity(signature, vector_db)
            # score = (euclidean_score * 2 + cosine_score) / 3
            # id_temp = scored_point.payload.get("id")
            score = scored_point.score
            if score < threshold:
                threshold = score
                result["id"] = scored_point.payload.get("id")

    return result

# Route to handle timekeeping
@app.route('/timekeeping', methods=['POST'])
# @jwt_required()
def timekeeping():
    # Nhận tệp hình ảnh từ yêu cầu
    image_file = request.files['image']
    type_action = request.form['type']
    collection = request.form['collection']

    # Đọc dữ liệu từ tệp ảnh
    image_data = image_file.read()

    # Tạo đối tượng hình ảnh từ dữ liệu byte
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt trong frame
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Gửi các tác vụ xử lý song song cho từng khuôn mặt
    future_results = []
    for face_location in face_locations:
        future = executor.submit(process_face_for_location, face_location, rgb_small_frame, collection)
        future_results.append(future)

    # Đợi kết quả trả về từ tất cả các tác vụ
    face_names = []
    face_labels = []
    for future in future_results:
        result = future.result()
        face_names.append(result["id"])
        face_labels.append(result["label"])

    
    time_in_hours = []
    timekeeping_time = datetime.now().time()
    # Ghi lại thông tin chấm công nếu khuôn mặt được nhận diện
    for name, label in zip(face_names, face_labels):
        if name != "Unknown" and label != "fake":
            try:
                # Kiểm tra nếu đã có bản ghi cho nhân viên và type_action cùng ngày
                record = Timekeeping.query.filter_by(employeeID=name, date=date.today(), status=type_action).first()
                if not record:
                    # Nếu không có bản ghi nào, thêm bản ghi mới
                    new_record = Timekeeping(
                        employeeID=name,
                        date=date.today(),
                        time=timekeeping_time,
                        status=type_action
                    )
                    db.session.add(new_record)
                    db.session.commit()
                    time_in_hours.append(timekeeping_time.strftime("%H:%M:%S"))
                else:
                    time = record.time
                    time_in_hours.append(time.strftime("%H:%M:%S"))

            except Exception as e:
                print(f'error: {str(e)}')
                return jsonify({'error': str(e)}), 500
            
        else:
            time_in_hours.append("")

    # Trả về phản hồi
    return jsonify({'face_names': face_names, 'locations': face_locations, 'time': time_in_hours})


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y2) ** 2)

def check_face_direction(left, right, nose):
    A = euclidean_distance(nose, right)
    B = euclidean_distance(left, right)
    turn_direction = A / B
    
    if turn_direction < 0.3:
        return "turn left"
    elif turn_direction > 0.7:
        return "turn right"
    elif turn_direction > 0.45 and turn_direction < 0.55:
        return "straight"

def find_largest_face(face_locations):
    if not face_locations:
        return None
    largest_face_index = np.argmax([(bottom - top) * (right - left) for (top, right, bottom, left) in face_locations])
    return largest_face_index

def create_embedding(employee_id, company):
    embeddings = []
    print(f"employee_id: {employee_id}")
    print(f"company: {company}")
    try:
        for file_name in os.listdir(f"person_temp/{employee_id}"):
            file_path = os.path.join(f"person_temp/{employee_id}", file_name)
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Tệp không tìm thấy: {file_path}")
                    raise FileNotFoundError(f"Tệp không tìm thấy: {file_path}")

                img = cv2.resize(img, (160, 160))
                img_array = np.expand_dims(img, axis=0)

                embedding_FaceNet = MyFaceNet.embeddings(img_array)
                embedding_FaceNet = embedding_FaceNet.flatten()
                embeddings.append(embedding_FaceNet)

        avg_embedding = np.mean(embeddings, axis=0)
        client = QdrantClient(url="http://localhost:6333")
        operation_info = client.upsert(
            collection_name=company,
            points=[
                PointStruct(id=str(uuid.uuid4()), vector=avg_embedding, payload={"id": employee_id})
            ]
        )
        print(operation_info) 
        return "Done"
    except:
        return "Error"


#######################################################################################
# Thêm nhân viên
@app.route('/add_person', methods=['POST'])
@jwt_required()
def add_image():
    identity = get_jwt_identity()
    role = identity['role']
    company = identity['company']

    if role != 'admin':
        return jsonify({'error': 'Access forbidden: Admins only'}), 403

    if request.method == 'POST':
        try:
            image_file = request.files['image']

            # Nhận các dữ liệu khác từ form
            employee_id = request.form.get('ID')
            name = request.form.get('name')
            birthday = request.form.get('birthday')
            gender = request.form.get('gender')
            address = request.form.get('address')
            phone = request.form.get('phone')
            action = request.form.get('action')

            # Kiểm tra nhân viên có tồn tại hay không
            if action == "check":
                employee = Employee.query.filter_by(employeeID=employee_id).first()

                if employee:
                    return jsonify({
                        'status': 'Exists',
                        'error': 'Employee already exists'
                    }), 401
                else:
                    action = "turn left"
            
            if action == "Done":
                try:
                    # Thêm employee mới vào cơ sở dữ liệu
                    new_employee = Employee(
                        employeeID=employee_id, name=name, birthday=birthday, gender=gender, 
                        address=address, phone=phone, company=company
                    )
                    db.session.add(new_employee)
                    print("Đã thêm employee mới vào cơ sở dữ liệu")
                    db.session.commit()
                    status = create_embedding(employee_id, company)

                    if status == "Error":
                        print("Lỗi khi tạo vector")
                        db.session.rollback()
                        raise Exception("Embedding creation failed")

                except Exception as e:
                    db.session.rollback()
                    return jsonify({"error": str(e)}), 500          

                return jsonify({'locations': None, 'status': status})
            

            try:
                image_data = image_file.read()
                image = Image.open(BytesIO(image_data))

            except Exception as e:
                return jsonify({'error': f'Dữ liệu hình ảnh không hợp lệ: {str(e)}'}), 400

            image = np.array(image)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(rgb_small_frame)

            largest_face_index = find_largest_face(face_locations)

            if largest_face_index is not None:
                (top, right, bottom, left) = face_locations[largest_face_index]
                face = rgb_frame[top * 4:bottom * 4, left * 4:right * 4]

                face_landmarks = face_recognition.face_landmarks(rgb_small_frame)
                face_landmark = face_landmarks[largest_face_index]

                nose_bridge = face_landmark['nose_bridge'][3]
                chin_left = face_landmark['chin'][2]
                chin_right = face_landmark['chin'][14]
                face_direction = check_face_direction(chin_left, chin_right, nose_bridge)

                if not os.path.exists(f"person_temp/{employee_id}"):
                    os.mkdir(f"person_temp/{employee_id}")

                if action == face_direction:
                    cv2.imwrite(f"person_temp/{employee_id}/{employee_id}_{face_direction}.jpg", face)
                    return jsonify({'locations': face_locations[largest_face_index], 'status': face_direction})
                else:
                    return jsonify({'locations': face_locations[largest_face_index], 'status': face_direction})
            else:
                return jsonify({'locations': [], 'status': None})

        except Exception as e:
            logging.error(f'Lỗi: {str(e)}', exc_info=True)
            return jsonify({'error': f'Lỗi: {str(e)}'}), 501

    return jsonify({'error': 'Phương thức yêu cầu không hợp lệ'}), 405


#######################################################################################
# Xuất dữ liệu 
@app.route('/data_export', methods=['GET'])
@jwt_required()  # Yêu cầu xác thực JWT
def get_timekeeping():
    identity = get_jwt_identity()
    role = identity['role']
    company = identity['company']
    option = request.form.get('option')
    date = request.form.get('date')
    month = request.form.get('month')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    if not company:
        return jsonify({"error": "Thiếu tham số company"}), 400

    query = db.session.query(Timekeeping).join(Employee).filter(Employee.company == company)

    try:
        if option == "date":
            # Truy vấn theo ngày cụ thể
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            query = query.filter(Timekeeping.date == date_obj)
        elif option == 'month':
            # Truy vấn theo tháng cụ thể
            month_obj = datetime.strptime(month, '%Y-%m').date()
            query = query.filter(
                Timekeeping.date >= month_obj,
                Timekeeping.date < (month_obj.replace(day=1) + timedelta(days=32)).replace(day=1)
            )
        elif option == "period":
            # Truy vấn theo khoảng thời gian
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            query = query.filter(Timekeeping.date >= start_date_obj, Timekeeping.date <= end_date_obj)
        else:
            return jsonify({"error": "Thiếu tham số thời gian hợp lệ (date, month, hoặc start_date và end_date)"}), 400
    except ValueError:
        return jsonify({"error": "Định dạng ngày hoặc tháng không hợp lệ"}), 400

    results = query.all()

    # Chuyển đổi kết quả thành định dạng JSON
    timekeeping_list = [
        {
            "id": tk.id,
            "employeeID": tk.employeeID,
            "date": tk.date.strftime('%Y-%m-%d'),
            "time": tk.time.strftime('%H:%M:%S'),
            "status": tk.status
        } for tk in results
    ]

    return jsonify(timekeeping_list)


#######################################################################################
@app.route('/delete_account', methods=['DELETE'])
@jwt_required()  # Yêu cầu xác thực JWT
def delete_account():
    identity = get_jwt_identity()
    company = identity['company']

    try:
        try: 
            # Delete the account with the specified company
            Account.query.filter_by(company=company).delete()
            db.session.commit()
        except: 
            db.session.rollback()
            logging.error(f'Lỗi: {str(e)}', exc_info=True)
            return jsonify({'error': f'Lỗi: {str(e)}'}), 501
        client = QdrantClient(url="http://localhost:6333")
        client.delete_collection(collection_name=company)
    except Exception as e:
        db.session.rollback()
        logging.error(f'Lỗi: {str(e)}', exc_info=True)
        return jsonify({'error': f'Lỗi: {str(e)}'}), 502
    finally:
        return jsonify({"Message": "Done"})


#######################################################################################
# Tìm thông tin nhân viên    
@app.route('/search_person', methods=['GET'])
@jwt_required()  # Yêu cầu xác thực JWT
def search_person():
    identity = get_jwt_identity()
    company = identity['company']
    employee_id = request.form.get('employeeID') 
    print(employee_id)

    record = Employee.query.filter_by(employeeID=employee_id, company=company).first()
    if record:
        employee = {
            'name': record.name,
            'birthday': record.birthday,
            'gender': record.gender,
            'address': record.address,
            'phone': record.phone
        }
        return jsonify(employee)
    else:
        return jsonify("Error", "Không tìm thấy nhân viên"), 501


#######################################################################################    
# Xóa nhân viên
@app.route('/delete_person', methods=['DELETE'])
@jwt_required()  # Yêu cầu xác thực JWT
def delete_person():
    identity = get_jwt_identity()
    company = identity['company']
    employee_id = request.form.get('employeeID') 
    
    try: 
        employee = Employee.query.filter_by(employeeID=employee_id).first()
        if employee:
            db.session.delete(employee)
            db.session.commit()
        else:
            return jsonify("Error", "Không tồn tại nhân viên"), 501
        try:
            client = QdrantClient(url="http://localhost:6333")

            client.delete(
                collection_name=company,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="id",
                                match=models.MatchValue(value=employee_id),
                            ),
                        ],
                    )
                ),
            )
        except:
            db.session.rollback()
            return jsonify("Error", "Có lỗi khi kết nối với vector database"), 502
    except:
        return jsonify("Error", "Có lỗi khi thao tác với cơ sở dữ liệu"), 503
    finally:
        return jsonify("Status", "Đã xóa nhân viên"), 200



@socketio.on('connect') 
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app)
