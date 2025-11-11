import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
def draw_lips_only(frame_rgb, results):
    annotated = frame_rgb.copy()

    if not results.multi_face_landmarks:
        return annotated

    lip_connections = mp_face_mesh.FACEMESH_LIPS
    lip_indices = {index for connection in lip_connections for index in connection}

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=lip_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

        h, w, _ = annotated.shape
        for idx in lip_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

    return annotated


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("✅ Kamera aktif — Tekan 'q' untuk keluar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Gagal membaca frame dari kamera")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            annotated = draw_lips_only(frame_rgb, results)
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

            cv2.imshow("Lip Landmarks - Real Time", annotated_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
