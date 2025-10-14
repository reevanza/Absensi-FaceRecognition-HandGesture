import cv2
import mediapipe as mp

#HAND GESTURE
#inisiasi mediapipe hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

#mengenali gesture tangan
def recognize_gesture(hand_landmarks):
    #ambil posisi ujung jari
    ujung_jempol = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ujung_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ujung_tengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ujung_manis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ujung_kelingking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    pangkal_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pangkal_tengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    pangkal_manis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pangkal_kelingking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    #Deteksi gesture
    # Thumbs Up 👍
    if (ujung_jempol.y < ujung_telunjuk.y and
        ujung_jempol.y < ujung_tengah.y and
        ujung_jempol.y < ujung_manis.y and
        ujung_jempol.y < ujung_kelingking.y):
        return "Thumbs Up"

    # Peace ✌️
    elif (ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_tengah.y < pangkal_tengah.y and
        ujung_manis.y > pangkal_manis.y and
        ujung_kelingking.y > pangkal_kelingking.y):
        return "Peace"

    # Rock 🤘
    elif (ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_kelingking.y < pangkal_kelingking.y and
        ujung_tengah.y > pangkal_tengah.y and
        ujung_manis.y > pangkal_manis.y):
        return "Rock"

    # OK 👌
    elif (abs(ujung_jempol.x - ujung_telunjuk.x) < 0.05 and
        abs(ujung_jempol.y - ujung_telunjuk.y) < 0.05 and
        ujung_tengah.y < pangkal_tengah.y and
        ujung_manis.y < pangkal_manis.y and
        ujung_kelingking.y < pangkal_kelingking.y):
        return "OK"

    # Tidak dikenali
    return "Gesture tidak diketahui"





#deteksi tangan
def detect_hand_gesture(image, hand):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks)
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, gesture, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), cv2.LINE_4)

    return image


#open cam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

while (cap.isOpened):
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    frame = detect_hand_gesture(frame, hands)
    #show frame
    cv2.imshow("HandGesture",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()