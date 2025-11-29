import numpy as np

class Kalman2D:
    def __init__(self, dt=1):
        # Estado: [x, y, vx, vy]^T → posición y velocidad
        self.x = np.zeros((4, 1), dtype=np.float32)

        # Matriz de transición del estado
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ], dtype=np.float32)

        # Matriz de observación (solo medimos posición)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Covarianza inicial
        self.P = np.eye(4, dtype=np.float32) * 500

        # Ruido del proceso (modelo interno)
        self.Q = np.eye(4, dtype=np.float32) * 0.01

        # Ruido de medición (sensor o YOLO)
        self.R = np.eye(2, dtype=np.float32) * 5

        # Matriz identidad
        self.I = np.eye(4, dtype=np.float32)

    def predecir(self):
        # Predicción del siguiente estado
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def corregir(self, z):
        """
        Corrige con una medición z = [[x], [y]]
        """
        y = z - self.H @ self.x  # innovación
        S = self.H @ self.P @ self.H.T + self.R  # covarianza innovación
        K = self.P @ self.H.T @ np.linalg.inv(S)  # ganancia de Kalman

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
