import numpy as np
import face_recognition as fr
import cv2
from playsound import playsound
video_capture = cv2.VideoCapture(0)

bruno_image = fr.load_image_file("images/MYPHOTO.jpg")
papa_image = fr.load_image_file("images/papa.jpg")
bhola_image = fr.load_image_file("images/Bhola.jpg")
bruno_face_encoding = fr.face_encodings(bruno_image)[0]
papa_face_encoding = fr.face_encodings(papa_image)[0]
bhola_face_encoding = fr.face_encodings(bhola_image)[0]

known_face_encondings = [bruno_face_encoding]
known_face_encondings = [papa_face_encoding]
known_face_encondings = [bhola_face_encoding]
known_face_names = ["Rahul","Papa","Bhola"]


while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Chor HAI!"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            playsound('alertsound.wav',False)
            print("GHAR ME CHOR BE ALERT !")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()