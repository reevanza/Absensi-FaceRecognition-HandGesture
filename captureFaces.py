import cv2
import os


person_name = input("Masukkan nama orang: ").strip()
dataset_dir = "dataset"
person_dir = os.path.join(dataset_dir, person_name)
os.makedirs(person_dir, exist_ok=True)


cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print("[INFO] Mulai ambil gambar. Tekan 'q' untuk berhenti.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{person_dir}/{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} #{count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Capture Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

print(f"[INFO] Pengambilan selesai, total {count} gambar disimpan di {person_dir}")
cam.release()
cv2.destroyAllWindows()
