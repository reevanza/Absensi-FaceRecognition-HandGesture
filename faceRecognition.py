import cv2
import dlib
import math

def euclidean_dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye_points, landmarks):
    p1 = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    p2 = (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y)
    p3 = (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y)
    p4 = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
    p5 = (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y)
    p6 = (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)

    ear = (euclidean_dist(p2, p6) + euclidean_dist(p3, p5)) / (2.0 * euclidean_dist(p1, p4))
    return ear

def mouth_aspect_ratio(landmarks):
    p48 = (landmarks.part(48).x, landmarks.part(48).y)
    p54 = (landmarks.part(54).x, landmarks.part(54).y)
    p51 = (landmarks.part(51).x, landmarks.part(51).y)
    p57 = (landmarks.part(57).x, landmarks.part(57).y)
    p49 = (landmarks.part(49).x, landmarks.part(49).y)
    p55 = (landmarks.part(55).x, landmarks.part(55).y)
    p53 = (landmarks.part(53).x, landmarks.part(53).y)
    p59 = (landmarks.part(59).x, landmarks.part(59).y)

    mar = (euclidean_dist(p51, p57) + euclidean_dist(p53, p59) + euclidean_dist(p49, p55)) / (3 * euclidean_dist(p48, p54))
    return mar

def smile_aspect_ratio(landmarks):
    p48 = (landmarks.part(48).x, landmarks.part(48).y)
    p54 = (landmarks.part(54).x, landmarks.part(54).y)
    p51 = (landmarks.part(51).x, landmarks.part(51).y)
    p57 = (landmarks.part(57).x, landmarks.part(57).y)

    sar = euclidean_dist(p48, p54) / euclidean_dist(p51, p57)
    return sar


# --- Load model dan kamera ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

# --- Thresholds ---
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.6
SMILE_AR_THRESH = 1.8  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)


        leftEAR = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmarks)
        rightEAR = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmarks)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(landmarks)
        sar = smile_aspect_ratio(landmarks)

        # Tampilkan status di layar
        y_offset = 60
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Kedip!", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Mulut terbuka!", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            y_offset += 40
        if sar > SMILE_AR_THRESH:
            cv2.putText(frame, "Senyum!", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gambar landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    cv2.imshow("Deteksi Ekspresi: Kedip, Mulut, Senyum", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
