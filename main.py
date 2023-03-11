import cv2
import mediapipe as mp

# Importar utilidades de dibujo y la clase FaceMesh de Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar la clase FaceMesh de Mediapipe con los valores de confianza mínima
# para la detección y el seguimiento
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # Leer un fotograma del video
        success, image = cap.read()
        if not success:
            print("No se puede obtener el fotograma de la cámara.")
            break
        
        # Convertir la imagen a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen con Mediapipe
        results = face_mesh.process(image)
        
        # Convertir la imagen de nuevo a BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dibujar puntos en la imagen
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(landmarks.landmark):
                    # Obtener las coordenadas x e y de cada punto de la malla
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    
                    # Dibujar un círculo en cada punto de la malla
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                
                # Dibujar la malla facial en la imagen
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))
        
        # Mostrar la imagen
        cv2.imshow("Malla facial", image)
        
        # Salir del bucle si se presiona la tecla Esc
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Liberar los recursos de la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
