import cv2
import os
import time

# --- Konfigurasi ---
DATASET_DIR = "dataset"
IMG_SIZE = (200, 200) # Ukuran gambar yang akan disimpan
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

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
cap.set(3, 1280) # Lebar
cap.set(4, 720) # Tinggi

print("\n[INSTRUKSI]")
print(f"- Target: Minimal 50 gambar dengan berbagai ekspresi dan sudut. (Tersimpan saat ini: {count})")
print("- Tekan **'SPACE'** untuk ambil gambar, **'Q'** untuk keluar")

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
        scaleFactor=1.1, # Sesuaikan jika deteksi sulit
        minNeighbors=5, 
        minSize=(100, 100)
    )

    detected_face_roi = None
    
    # Ambil wajah terbesar (asumsi hanya 1 orang)
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        # Gambar kotak di frame display
        cv2.rectangle(frame_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_display, f"Wajah Ditemukan ({w}x{h})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Potong dan ubah ukuran wajah untuk disimpan
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size > 0:
            detected_face_roi = cv2.resize(face_roi, IMG_SIZE)
            
            # Tampilkan ROI yang akan disimpan di sudut
            cv2.putText(frame_display, "Preview Disimpan:", (frame.shape[1] - 250, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_display[10:10+IMG_SIZE[1], frame.shape[1]-10-IMG_SIZE[0]:frame.shape[1]-10] = cv2.cvtColor(detected_face_roi, cv2.COLOR_GRAY2BGR)


    # Tampilkan informasi
    cv2.putText(frame_display, f"Gambar: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Capture Dataset (Wajah Ditemukan)", frame_display)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        if detected_face_roi is not None:
            count += 1
            filename = os.path.join(person_path, f"{count}.jpg")
            
            # Simpan wajah yang sudah diproses (grayscale, 200x200)
            cv2.imwrite(filename, detected_face_roi) 
            print(f"[SUCCESS] Gambar {count} tersimpan.")
            # Beri jeda visual setelah mengambil gambar
            cv2.putText(frame_display, "SAVED!", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.imshow("Capture Dataset (Wajah Ditemukan)", frame_display)
            cv2.waitKey(200) # Tahan 200ms
        else:
            print("[WARNING] Wajah tidak terdeteksi. Posisikan wajah ke tengah frame.")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Total {count} gambar tersimpan untuk {person_name}")