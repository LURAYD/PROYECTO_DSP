import cv2
import numpy as np
from collections import deque

class FiltroKalmanIndividual:
    def __init__(self, id_inicial, punto_inicial):
        self.id = id_inicial
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kf.statePost = np.array([[np.float32(punto_inicial[0])],
                                      [np.float32(punto_inicial[1])],
                                      [0], [0]], np.float32)

        self.age = 0
        self.missing_count = 0
        self.prediccion = punto_inicial
        self.trayectoria = deque(maxlen=20)
        self.trayectoria.append(punto_inicial)

    def predecir(self):
        p = self.kf.predict()
        self.age += 1
        self.prediccion = (int(p[0]), int(p[1]))
        self.trayectoria.append(self.prediccion)
        return self.prediccion

    def corregir(self, centro_medido):
        medicion = np.array([[np.float32(centro_medido[0])],
                             [np.float32(centro_medido[1])]], np.float32)
        self.kf.correct(medicion)
        self.missing_count = 0

    def predecir_futuro(self, pasos=5):
        pred = self.kf.statePost.copy()
        future_points = []
        for _ in range(pasos):
            pred = np.dot(self.kf.transitionMatrix, pred)
            punto = (int(pred[0]), int(pred[1]))
            future_points.append(punto)
        return future_points


class GestorRastreo:
    def __init__(self, distancia_maxima=80, max_frames_perdidos=15):
        self.trackers = []
        self.next_id = 1
        self.dist_threshold = distancia_maxima
        self.max_missing = max_frames_perdidos

    def actualizar(self, detecciones_yolo):
        for t in self.trackers:
            t.predecir()

        asignaciones = []
        detecciones_pendientes = list(range(len(detecciones_yolo)))
        trackers_pendientes = list(range(len(self.trackers)))

        if len(self.trackers) > 0 and len(detecciones_yolo) > 0:
            for i, det in enumerate(detecciones_yolo):
                mejor_dist = self.dist_threshold
                mejor_tracker_idx = -1

                for j, trk in enumerate(self.trackers):
                    if j not in trackers_pendientes:
                        continue
                    d = np.linalg.norm(np.array(det) - np.array(trk.prediccion))
                    if d < mejor_dist:
                        mejor_dist = d
                        mejor_tracker_idx = j

                if mejor_tracker_idx != -1:
                    asignaciones.append((mejor_tracker_idx, i))
                    trackers_pendientes.remove(mejor_tracker_idx)
                    detecciones_pendientes.remove(i)

        for trk_idx in trackers_pendientes:
            self.trackers[trk_idx].missing_count += 1

        for trk_idx, det_idx in asignaciones:
            self.trackers[trk_idx].corregir(detecciones_yolo[det_idx])

        for det_idx in detecciones_pendientes:
            nuevo_trk = FiltroKalmanIndividual(self.next_id, detecciones_yolo[det_idx])
            self.trackers.append(nuevo_trk)
            self.next_id += 1

        self.trackers = [t for t in self.trackers if t.missing_count <= self.max_missing]
        return self.trackers
