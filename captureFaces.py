import cv2
import os
import time

# --- Konfigurasi ---
DATASET_DIR = "dataset"
IMG_SIZE = (200, 200) # Ukuran gambar yang akan disimpan (200x200)
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

# --- Timer Otomatis & Batas Target ---
CAPTURE_INTERVAL = 0.3  # LEBIH CEPAT: Ambil gambar setiap 0.3 detik
MIN_REQUIRED_IMAGES = 70 # BERHENTI OTOMATIS: Target 70 gambar
last_capture_time = time.time()

# --- Persiapan Folder ---
person_name = input("Masukkan nama orang: ").strip()
if not person_name:
    print("[ERROR] Nama tidak boleh kosong.")
    exit()

person_path = os.path.join(DATASET_DIR, person_name)
os.makedirs(person_path, exist_ok=True)

# Cari nomor gambar terakhir
existing_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
count = len(existing_files)

print(f"[INFO] Folder '{person_name}' siap. Gambar akan dimulai dari nomor {count + 1}.")

# --- Inisialisasi Kamera ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720) 

print("\n[INSTRUKSI]")
print(f"- **TARGET:** Kumpulkan {MIN_REQUIRED_IMAGES} gambar. (Otomatis berhenti jika tercapai).")
print(f"- Proses pengambilan gambar wajah berlangsung secara **Otomatis** setiap {CAPTURE_INTERVAL} detik.")
print("- Lakukan berbagai ekspresi (senyum, mulut terbuka) dan sudut wajah.")
print("- Tekan **'Q'** untuk keluar\n")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_display = frame.copy()
    gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
    
    # Deteksi Wajah
    faces = face_detector.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100)
    )

    detected_face_roi = None
    face_detected_flag = False
    
    if len(faces) > 0:
        face_detected_flag = True
        # Ambil wajah terbesar (asumsi hanya 1 orang)
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        # Gambar kotak di frame display
        # Warna kotak berubah saat target tercapai
        box_color = (0, 255, 0) if count < MIN_REQUIRED_IMAGES else (255, 165, 0)
        cv2.rectangle(frame_display, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame_display, "Wajah Terdeteksi", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Potong dan ubah ukuran wajah untuk disimpan
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size > 0:
            detected_face_roi = cv2.resize(face_roi, IMG_SIZE)
            
            # Tampilkan ROI yang akan disimpan di sudut
            cv2.putText(frame_display, "Preview:", (frame.shape[1] - 250, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_display[10:10+IMG_SIZE[1], frame.shape[1]-10-IMG_SIZE[0]:frame.shape[1]-10] = cv2.cvtColor(detected_face_roi, cv2.COLOR_GRAY2BGR)
            
    else:
        # Jika wajah tidak terdeteksi
        cv2.putText(frame_display, "!! Wajah tidak terdeteksi !!", (frame.shape[1] // 2 - 150, frame.shape[0] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # --- Logika Otomatis Pengambilan Gambar & Batas Berhenti ---
    current_time = time.time()
    
    if count >= MIN_REQUIRED_IMAGES:
        # Jika target sudah tercapai, tampilkan pesan berhenti
        cv2.putText(frame_display, f"✅ TARGET ({MIN_REQUIRED_IMAGES}) TERCAPAI! Tekan 'Q' untuk keluar", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
    elif face_detected_flag and detected_face_roi is not None and (current_time - last_capture_time) > CAPTURE_INTERVAL:
        
        # Logika Simpan
        count += 1
        filename = os.path.join(person_path, f"{count}.jpg")
        
        cv2.imwrite(filename, detected_face_roi) 
        print(f"[SUCCESS] Gambar {count} tersimpan.")
        last_capture_time = current_time # Reset timer
        
        # Feedback Visual
        cv2.putText(frame_display, "SAVED!", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        
    # Tampilkan informasi status
    status_text = f"Gambar: {count} / {MIN_REQUIRED_IMAGES}"
    status_color = (0, 255, 0) if count >= MIN_REQUIRED_IMAGES else (0, 165, 255)
    
    cv2.putText(frame_display, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    cv2.imshow("Capture Dataset (Otomatis)", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Total {count} gambar tersimpan untuk {person_name}")