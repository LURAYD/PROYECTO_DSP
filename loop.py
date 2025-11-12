import subprocess
import os

video_entrada = "video1.mp4"
video_salida = "video_4h.mp4"
duracion_deseada_min = 240  # 4 horas

# Detectar duración con ffprobe
cmd = [
    "ffprobe", "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    video_entrada
]

try:
    output = subprocess.check_output(cmd).decode().strip()
    duracion_seg = float(output)
    duracion_min = duracion_seg / 60
    repeticiones = int(duracion_deseada_min / duracion_min) - 1

    print(f"🎬 Duración del video original: {duracion_min:.2f} min")
    print(f"🔁 Repeticiones necesarias: {repeticiones + 1}")

    # Crear el video largo
    comando_ffmpeg = [
        "ffmpeg",
        "-stream_loop", str(repeticiones),
        "-i", video_entrada,
        "-c", "copy",
        video_salida
    ]

    subprocess.run(comando_ffmpeg)
    print(f"✅ Video generado: {video_salida}")

except Exception as e:
    print("❌ Error detectando la duración o generando el video:")
    print(e)
