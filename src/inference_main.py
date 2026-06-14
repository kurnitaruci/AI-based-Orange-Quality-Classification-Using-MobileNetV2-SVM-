import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from rpi_lcd import LCD
import csv
import os
import joblib
import numpy as np
import cv2
import tensorflow as tf


from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
STEP_90 = 200

# Inisialisasi LCD
lcd = LCD()

# Konfigurasi Pin GPIO
IR_PIN = 25
DIR_PIN = 26
STEP_PIN = 13
SERVO1_PIN = 19
SERVO2_PIN = 12
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.IN)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT)
GPIO.setup(SERVO1_PIN, GPIO.OUT)
GPIO.setup(SERVO2_PIN, GPIO.OUT)

servo1 = GPIO.PWM(SERVO1_PIN, 50)  # 50 Hz servo 1
servo2 = GPIO.PWM(SERVO2_PIN, 50)  # 50 Hz servo 2
servo1.start(0)
servo2.start(0)

# model_path = '/home/piskripsi/Documents/skripsi_tensor/best_cls_2.onnx'
# session = onnxruntime.InferenceSession(model_path)
# input_name = session.get_inputs()[0].name
feature_extract = load_model("/home/piskripsi/RUCI/feature_extractor_tes_baruv1_new.h5")

# PCA + SVM + label encoder classes
bundle = joblib.load("/home/piskripsi/RUCI/model_tes_baruv1.pkl")
scaler = bundle["scaler"]
pca = bundle["pca"]
classifier = bundle["classifier"]
classes = bundle["classes"]

csv_log_path = '/home/piskripsi/RUCI/testing_log_ruci_test.csv'

STEP_90 = 200

def init_csv_log():
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Class', 'Confidence', 'Inference_Time(s)', 'Total_System_Time(s)'])

def append_csv_log(class_name, confidence, inference_time, total_system_time):
    with open(csv_log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, class_name, f"{confidence:.2f}", f"{inference_time:.4f}", f"{total_system_time:.4f}"])
 

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)        # WAJIB! bukan /255
    img = np.expand_dims(img, axis=0)
    return img



def get_stable_frame(cap, num_discard=5):
    for _ in range(num_discard):
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame awal dari webcam")
            return None
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame untuk inferensi")
        return None
    return frame


def predict_frame(frame):
    img = preprocess_frame(frame)

    features = feature_extract.predict(img, verbose=0)
    
    features_scaled = scaler.transform(features)

    features_pca = pca.transform(features_scaled)

    probs = classifier.predict_proba(features_pca)[0]

    pred = np.argmax(probs)
    confidence = probs[pred]

    return pred, classes[pred], confidence


def move_stepper(direction, steps, delay=0.002):
    GPIO.output(DIR_PIN, direction)
    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)

def go_and_back(step_target):
    move_stepper(GPIO.HIGH, step_target)
    time.sleep(1)
    move_stepper(GPIO.LOW, step_target)

        
# === Fungsi Kontrol Servo (Tidak Berubah) ===
def set_servo_angle(servo, angle):
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

def action_based_on_class(class_index):
    print(f"Eksekusi aksi untuk kelas: {classes[class_index]}")

    if class_index == 0:      # GRADE A
        step_target = 0
        direction = None

    elif class_index == 1:    # GRADE B
        step_target = STEP_90
        direction = GPIO.HIGH

    elif class_index == 2:    # GRADE DEFECT
        step_target = STEP_90 * 2
        direction = GPIO.HIGH

    elif class_index == 3:    # GRADE SUPER
        step_target = STEP_90
        direction = GPIO.LOW   

    else:
        print("Kelas tidak dikenali")
        return

    # Gerakkan stepper jika perlu
    if step_target > 0 and direction is not None:
        move_stepper(direction, step_target)

    # Aksi servo (tetap)
    set_servo_angle(servo1, 0)
    set_servo_angle(servo2, 45)
    time.sleep(4)
    set_servo_angle(servo1, 45)
    set_servo_angle(servo2, 0)

    # Kembalikan stepper ke posisi awal
    if step_target > 0 and direction is not None:
        # balik arah
        back_direction = GPIO.LOW if direction == GPIO.HIGH else GPIO.HIGH
        move_stepper(back_direction, step_target)

def main():
    lcd.text(f"Sistem Pemilah jeruk Keprok", 1)
    time.sleep(1)
    lcd.text(f"Mempersiapkan sistem...", 1)
    time.sleep(2)

    init_csv_log()

    lcd.text(f"Mempersiapkan Kamera...", 1)
    time.sleep(2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka webcam")
        lcd.text(f"Kamera Error, tolong periksa!", 1)
        return

    print("Sistem siap. Tunggu sensor IR aktif untuk inferensi...")

    try:
        while True:
            lcd.text(f"Sistem siap!", 1)
            lcd.text(f"Menunggu input...", 2)

            if GPIO.input(IR_PIN) == GPIO.LOW:
                time.sleep(3)
                system_time_start = time.time()
                frame = get_stable_frame(cap)
                if frame is None:
                    continue

                try:
                    input_tensor = preprocess_frame(frame)
                    print(f"----------------------------------")
                    print(f"Input tensor shape: {input_tensor.shape}")

                    start_time = time.time()
                    # outputs = session.run(None, {input_name: input_tensor})
                    class_name, confidence = predict_frame(frame)
                    inference_time = time.time() - start_time

                    # probabilities = outputs[0][0]
                    class_index = classes.index(class_name)
                    # confidence = probabilities[class_index]
                    
                    
                    print(f"===================================")
                    # print(f"Probabilitas kelas: {probabilities}")
                    print(f"Deteksi: {classes[class_index]} dengan confidence {confidence:.2f}")
                    print(f"Waktu Inferensi: {inference_time*1000:.2f} ms")
                    print(f"===================================")

                    lcd.text(f"{classes[class_index][:6]} {int(confidence*100)}%", 1)
                    lcd.text(f"Time:{int(inference_time*1000)}ms", 2)


                    action_based_on_class(class_index)
                    system_time_end = time.time()
                    total_system_time = system_time_end - system_time_start
                    append_csv_log(classes[class_index], confidence, inference_time, total_system_time)
                    
                    print(f"Waktu Total Sistem: {total_system_time:.2f} detik")
                    
                except Exception as e:
                    print(f"Error saat inferensi: {e}")

                # Debounce delay
                time.sleep(0.5)

            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Program dihentikan oleh pengguna.")

    finally:
        servo1.stop()
        servo2.stop()
        cap.release()
        GPIO.cleanup()
        lcd.clear()

if __name__ == "__main__":
    main()
