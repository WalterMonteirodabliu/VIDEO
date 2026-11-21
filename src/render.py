"""Renderiza os videos do telao. Uso: python3 render.py [saida_dir]"""
import os, subprocess, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
import telao as T

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T.HERE, "..", "output")
FULL_LEN = T.REVEAL_START + 6 * T.LOOP_T     # intro + 6 ciclos do loop (termina emendando no Loop)
A = None

def init():
    global A
    A = T.build_assets()

def full_frame(i):
    return T.to_u8(T.full_frame(A, i / T.FPS), i).tobytes()

def loop_frame(i):
    return T.to_u8(T.art_frame(A, i / T.FPS), i).tobytes()

def encode(path, fn, n):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{T.W}x{T.H}", "-r", str(T.FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init) as pool:
        for k, b in enumerate(pool.imap(fn, range(n), chunksize=4)):
            p.stdin.write(b)
            if k % 150 == 0:
                print(f"{os.path.basename(path)}: {k}/{n}", flush=True)
    p.stdin.close(); p.wait()

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    init()
    Image.fromarray(T.to_u8(T.art_frame(A, 0.0))).save(os.path.join(OUT, "Arte_Telao_1920x512.png"))
    encode(os.path.join(OUT, "Loop_Arte_1920x512.mp4"), loop_frame, int(T.LOOP_T * T.FPS))
    encode(os.path.join(OUT, "O_Autor_da_Historia_Telao_1920x512.mp4"), full_frame, int(round(FULL_LEN * T.FPS)))
