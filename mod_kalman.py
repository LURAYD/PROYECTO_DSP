# mod_kalman.py
import numpy as np
import time
from collections import deque
import math

class Kalman2D:
    def __init__(self, x, y, dt=1):
        self.x = np.array([[x], [y], [0.], [0.]], dtype=np.float32)

        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.P = np.eye(4, dtype=np.float32) * 500
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 25
        self.I = np.eye(4, dtype=np.float32)

        self.history = deque(maxlen=10)
        self.time_since_update = 0
        self.age = 0
        self.id = int(time.time() * 1000) % 100000

    def predecir(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.age += 1
        self.time_since_update += 1
        cx, cy = float(self.x[0]), float(self.x[1])
        self.history.append((cx, cy))
        return (cx, cy)

    def corregir(self, z):
        z = np.array(z).reshape((2,1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (self.I - K @ self.H) @ self.P
        self.time_since_update = 0

    def get_state(self):
        return self.x.flatten()


def distancia(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


class GestorKalman:
    def __init__(self, umbral_distancia=60, edad_maxima=10):
        self.trackers = []
        self.umbral = umbral_distancia
        self.edad_maxima = edad_maxima

    def actualizar(self, mediciones):
        asignados = set()
        asignacion_deteccion = [False]*len(mediciones)

        # Predicción de todos
        for t in self.trackers:
            t.predecir()

        # Asociación detección -> tracker
        for i, z in enumerate(mediciones):
            mejor_dist = float('inf')
            mejor_idx = -1
            for j, t in enumerate(self.trackers):
                if j in asignados:
                    continue
                pred = (t.x[0,0], t.x[1,0])
                d = distancia(pred, z)
                if d < mejor_dist and d < self.umbral:
                    mejor_dist = d
                    mejor_idx = j
            if mejor_idx != -1:
                self.trackers[mejor_idx].corregir(z)
                asignados.add(mejor_idx)
                asignacion_deteccion[i] = True

        # Nuevos objetos
        for i, z in enumerate(mediciones):
            if not asignacion_deteccion[i]:
                self.trackers.append(Kalman2D(z[0], z[1]))

        # Eliminar viejos
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.edad_maxima]

        return self.trackers
