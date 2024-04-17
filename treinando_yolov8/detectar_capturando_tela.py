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
window_title = "Queimadas na Amazônia chocam o mundo - YouTube e mais 3 páginas - Pessoal — Microsoft​ Edge"

# Encontra a janela desejada
window = find_window_by_title(window_title)
if window is None:
    print("Janela não encontrada:", window_title)
    exit()

# Obtém as coordenadas da janela
left, top, width, height = window.left, window.top, window.width, window.height

# Carrega o modelo YOLO treinado com Among
model = YOLO("fireAndSmokev2.pt")

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

# offset_x = 400 #0
# offset_y = 300 #30
# wincap = WindowCapture(size=(800, 600), origin=(offset_x, offset_y))

# # Usa modelo da Yolo
# # Model	    size    mAPval  Speed       Speed       params  FLOPs
# #           (pixels) 50-95  CPU ONNX A100 TensorRT   (M)     (B)
# #                           (ms)        (ms)
# # YOLOv8n	640	    37.3	80.4	    0.99	    3.2	    8.7
# # YOLOv8s	640	    44.9	128.4	    1.20	    11.2	28.6
# # YOLOv8m	640	    50.2	234.7	    1.83	    25.9	78.9
# # YOLOv8l	640	    52.9	375.2	    2.39	    43.7	165.2
# # YOLOv8x	640	    53.9	479.1	    3.53	    68.2	257.8

# # model = YOLO("yolov8n.pt")

# # Usa modelo treinado com Among
# model = YOLO("runs/detect/train9/weights/best.pt")

# track_history = defaultdict(lambda: [])
# seguir = True
# deixar_rastro = True

# while True:
#     img = wincap.get_screenshot()

#     if seguir:
#         results = model.track(img, persist=True)
#     else:
#         results = model(img)

#     # Process results list
#     for result in results:
#         # Visualize the results on the frame
#         img = result.plot()

#         if seguir and deixar_rastro:
#             try:
#                 # Get the boxes and track IDs
#                 boxes = result.boxes.xywh.cpu()
#                 track_ids = result.boxes.id.int().cpu().tolist()

#                 # Plot the tracks
#                 for box, track_id in zip(boxes, track_ids):
#                     x, y, w, h = box
#                     track = track_history[track_id]
#                     track.append((float(x), float(y)))  # x, y center point
#                     if len(track) > 30:  # retain 90 tracks for 90 frames
#                         track.pop(0)

#                     # Draw the tracking lines
#                     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                     cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
#             except:
#                 pass

#     cv2.imshow("Tela", img)

#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# cv2.destroyAllWindows()
# print("desligando")


