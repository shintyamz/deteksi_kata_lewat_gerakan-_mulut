import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================
# 🔹 Fungsi untuk menggambar landmark wajah
# ==========================
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])

        # Gambar jaring wajah (mesh)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Gambar kontur wajah
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )

        # Gambar iris mata
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image


# ==========================
# 🔹 Setup model Face Landmarker
# ==========================
model_path = os.path.join("models", "face_landmarker_v2_with_blendshapes.task")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ File model tidak ditemukan: {model_path}")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,  # ubah ke True kalau mau deteksi ekspresi
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.FaceLandmarker.create_from_options(options)

# ==========================
# 🔹 Jalankan kamera real-time
# ==========================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka")
    exit()

print("✅ Kamera aktif — Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame dari kamera")
        break

    # Ubah ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Deteksi wajah dari video
    detection_result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Gambar hasil landmark di frame kamera
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

    # Tampilkan hasil di jendela kamera
    cv2.imshow("Face Landmarks - Real Time", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan semua jendela
cap.release()
cv2.destroyAllWindows()
