import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pygetwindow as gw
import pyautogui

# Função para encontrar uma janela pelo título
def find_window_by_title(title):
    for window in gw.getAllWindows():
        print(window.title)
        if window.title == title:
            return window
    return None

# Nome da janela que você deseja capturar
window_title = "Novo separador - Google Chrome"

# Encontra a janela desejada
window = find_window_by_title(window_title)
if window is None:
    print("Janela não encontrada:", window_title)
    # model = YOLO("fireAndSmokev2.pt")
    model = YOLO("C:\\Users\\Joabe\\OneDrive\\Área de Trabalho\\fireguardian\\FireGuardVision\\runs\\detect\\train24\\weights\\best.pt")
# 
    

    video_path = "video.mp4"  # Altere para o caminho do seu vídeo
    track_history = defaultdict(lambda: [])
    seguir = True
    deixar_rastro = True

# Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if seguir:
            results = model.track(frame, persist=True)
        else:
            results = model(frame)

        # Processa os resultados
        for result in results:
            # Visualiza os resultados no frame
            frame = result.plot()

            if seguir and deixar_rastro:
                try:
                    # Obtém as caixas delimitadoras e IDs de rastreamento
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Desenha as linhas de rastreamento
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # ponto central x, y
                        if len(track) > 30:  # mantém 30 pontos de rastreamento para 30 quadros
                            track.pop(0)

                        # Desenha as linhas de rastreamento
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except:
                    pass

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
else:
# Obtém as coordenadas da janela
    left, top, width, height = window.left, window.top, window.width, window.height

    # Carrega o modelo YOLO treinado 
    model = YOLO("C:\\Users\\Joabe\\OneDrive\\Área de Trabalho\\fireguardian\\FireGuardVision\\runs\\detect\\train24\\weights\\best.pt")

    track_history = defaultdict(lambda: [])
    seguir = True
    deixar_rastro = True
    


while True:
    # Captura a área da janela
    frame = pyautogui.screenshot(region=(left, top, width, height))

    if seguir:
        results = model.track(frame, persist=True)
    else:
        results = model(frame)

    # Processa os resultados
    for result in results:
        # Visualiza os resultados no frame
        frame = result.plot()

        if seguir and deixar_rastro:
            try:
                # Obtém as caixas delimitadoras e IDs de rastreamento
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Desenha as linhas de rastreamento
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # ponto central x, y
                    if len(track) > 30:  # mantém 30 pontos de rastreamento para 30 quadros
                        track.pop(0)

                    # Desenha as linhas de rastreamento
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except:
                pass

    cv2.imshow("Window Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando...")
