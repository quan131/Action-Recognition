import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from CNN_LSTM import model
import time

frame_size=64
frame_len=5

# Khởi tạo video capture
cap = cv2.VideoCapture("input/ben.mp4")  # '0: Camera' or 'path to video'

# Khởi tạo font chữ và vị trí hiển thị
font = cv2.FONT_HERSHEY_SIMPLEX
text_pos = (10, 30)  # Ví dụ: vị trí góc trên bên trái

file_path = "classInd.txt"
# Đọc file txt vào DataFrame
label_data = pd.read_csv(file_path, sep=' ', header=None, names=['index', 'labels'])
# Lấy labels và bỏ vào mảng
label_data = label_data['labels'].tolist()

# Bộ đệm khung hình
frames = []

# Khởi tạo thời gian bắt đầu
start_time = time.time()

while True:
    # Chụp khung hình mới
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý khung hình
    frame_resized = cv2.resize(frame, (frame_size, frame_size))
    frame_cvt = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_cvt / 255.0
    frames.append(frame_normalized)

    frames = frames[-frame_len:]  # Duy trì độ dài chuỗi

    # Dự đoán hành động nếu đủ độ dài chuỗi
    if len(frames) == frame_len:
        input_frames = np.expand_dims(frames, axis=0)
        predicted_action = model.predict(input_frames)
        predicted_action_str = label_data[np.argmax(predicted_action[0])]
        cv2.putText(frame, f"Action: {predicted_action_str}", text_pos, font, 0.8, (0, 0, 255), 2)
        
        # Khởi tạo biến FPS
        fps = 1

        # Tính toán FPS trung bình mỗi giây
        elapsed_time = time.time() - start_time
        if elapsed_time:
            avg_fps = fps / elapsed_time
            fps = 0  # Khởi tạo lại biến FPS
            start_time = time.time()  # Cập nhật thời gian bắt đầu

            # Hiển thị FPS lên màn hình
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 50), font, 0.8, (0, 0, 255), 2)

    # Hiển thị khung hình và hành động dự đoán
    cv2.imshow('Action Recognition', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()