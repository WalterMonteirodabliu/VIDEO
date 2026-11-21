"""
O Autor da Historia - motion graphic para o telao (1920x512).

Camadas:
  - build_art()      -> arte estatica do telao (montada a partir do flyer)
  - art_frame(t)     -> arte animada em loop perfeito (periodo LOOP_T)
  - intro_frame(s)   -> introducao: estrela subindo sobre fundo chroma,
                        brilho, expansao e revelacao da arte
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 512
FPS = 30
LOOP_T = 10.0            # periodo do loop da arte (s)

# linha do tempo da introducao (s)
RISE_END = 3.0           # estrela termina de subir
SHINE_END = 4.0          # estrela brilha
REVEAL_START = SHINE_END # inicio da expansao / revelacao
REVEAL_END = 5.5         # arte totalmente revelada
INTRO_LEN = 6.0          # ate aqui ainda ha restos de brilho da estrela

CHROMA = np.array([0.0, 1.0, 0.0], np.float32)  # verde puro #00FF00
GOLD = np.array([1.0, 0.80, 0.52], np.float32)
WARM_WHITE = np.array([1.0, 0.95, 0.85], np.float32)

STAR_START = (960.0, 485.0)   # nasce no rodape e sobe
STAR_END = (1062.0, 70.0)   # topo do feixe de luz da arte

FONT_DIR = "/usr/share/fonts/opentype/inter"


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return t ** 3


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def resize(a, w, h):
    return np.asarray(to_img(a).resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0


def feather_mask(h, w, left, right, top, bottom):
    m = np.ones((h, w), np.float32)
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    if left:
        m *= _smooth(0, left, x)[None, :]
    if right:
        m *= _smooth(0, right, (w - 1) - x)[None, :]
    if top:
        m *= _smooth(0, top, y)[:, None]
    if bottom:
        m *= _smooth(0, bottom, (h - 1) - y)[:, None]
    return m


def paste(dst, src, mask, x0, y0):
    """Cola src (h,w,3) com mascara em dst nas coordenadas x0,y0 (recorta bordas)."""
    h, w = src.shape[:2]
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if dx1 <= dx0 or dy1 <= dy0:
        return
    s = src[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    m = mask[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0, None]
    dst[dy0:dy1, dx0:dx1] = dst[dy0:dy1, dx0:dx1] * (1 - m) + s * m


def text_layer(draw_items, size=(W, H)):
    """Desenha textos em uma camada RGBA e devolve (rgb, alpha)."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for txt, font, cx, cy, color in draw_items:
        bb = d.textbbox((0, 0), txt, font=font, anchor="mm")
        d.text((cx, cy), txt, font=font, fill=color, anchor="mm")
    a = np.asarray(img).astype(np.float32) / 255.0
    return a[..., :3], a[..., 3]


# ---------------------------------------------------------------------------
# Arte estatica
# ---------------------------------------------------------------------------
class Assets:
    pass


def build_assets():
    A = Assets()
    flyer = load(os.path.join(HERE, "flyer.jpg"))
    ref = load(os.path.join(HERE, "telao_ref.jpg"))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A.yy, A.xx = yy, xx

    # ---- fundo: escuro quente com halo suave e vinheta
    bg = np.zeros((H, W, 3), np.float32)
    base = np.array([0.020, 0.013, 0.007], np.float32)
    bg[:] = base
    halo = np.exp(-(((xx - 1000) / 900) ** 2 + ((yy - 200) / 420) ** 2))
    bg += halo[..., None] * np.array([0.045, 0.028, 0.012], np.float32)
    halo2 = np.exp(-(((xx - 300) / 420) ** 2 + ((yy - 330) / 300) ** 2))
    bg += halo2[..., None] * np.array([0.05, 0.030, 0.012], np.float32)

    art = bg.copy()

    # ---- presepio (lado esquerdo)
    nat = flyer[780:1400, 80:900]                 # 820 x 620
    nh = H + 8
    nw = int(round(nat.shape[1] * nh / nat.shape[0]))
    nat = resize(nat, nw, nh)
    nm = feather_mask(nh, nw, left=70, right=150, top=40, bottom=0)
    A.nat_x0 = -40
    paste(art, nat, nm, A.nat_x0, -4)
    # posicao do bebe (luz) na arte final
    sc = nh / 620.0
    A.baby = (A.nat_x0 + (430 - 80) * sc, -4 + (1145 - 780) * sc)

    # ---- titulo (centro) com o feixe de luz original do flyer
    ttl = flyer[240:770, 10:890]                   # 880 x 530
    th = H
    tw = int(round(ttl.shape[1] * th / ttl.shape[0]))
    ttl = resize(ttl, tw, th)
    tmask = feather_mask(th, tw, left=40, right=40, top=0, bottom=0)
    A.title_x0 = 960 - tw // 2
    paste(art, ttl, tmask, A.title_x0, 0)

    # mascara das letras douradas (para o brilho que passa pelo titulo)
    lum = ttl @ np.array([0.30, 0.59, 0.11], np.float32)
    warm = ttl[..., 0] - ttl[..., 2]
    letters = _smooth(0.42, 0.62, lum) * _smooth(0.10, 0.22, warm)
    # so a regiao do titulo "O Autor da Historia" (exclui data/subtitulo)
    ty = np.arange(th, dtype=np.float32)[:, None]
    letters *= (_smooth(70, 85, ty) * (1 - _smooth(405, 420, ty)))
    L = np.zeros((H, W), np.float32)
    L[:, A.title_x0:A.title_x0 + tw] = letters * tmask
    A.letters = L

    # pontos de brilho (glints) sobre as letras mais claras
    cand = np.argwhere((L > 0.9) & (lum_full(art) > 0.80))
    rng = np.random.default_rng(7)
    rng.shuffle(cand)
    picks = []
    for y, x in cand:
        if all((x - px) ** 2 + (y - py) ** 2 > 140 ** 2 for py, px in picks):
            picks.append((y, x))
        if len(picks) >= 6:
            break
    A.glints = [(float(x), float(y)) for y, x in picks]

    # ---- coluna da direita: logo + local
    logo = ref[418:490, 840:1080]
    la = _smooth(0.30, 0.85, logo[..., 0])        # branco sobre azul -> alfa
    lh, lw = la.shape
    s = 1.55
    la = np.asarray(Image.fromarray((la * 255).astype(np.uint8)).resize(
        (int(lw * s), int(lh * s)), Image.LANCZOS)).astype(np.float32) / 255.0
    lh, lw = la.shape
    grad = np.linspace(0, 1, lh, dtype=np.float32)[:, None, None]
    gold_top = np.array([0.93, 0.80, 0.58], np.float32)
    gold_bot = np.array([0.72, 0.53, 0.30], np.float32)
    lrgb = gold_top * (1 - grad) + gold_bot * grad
    lrgb = np.broadcast_to(lrgb, (lh, lw, 3)).copy()
    cx = 1652
    paste(art, lrgb, la, int(cx - lw / 2), 118)

    f_b = ImageFont.truetype(os.path.join(FONT_DIR, "Inter-Bold.otf"), 34)
    f_r = ImageFont.truetype(os.path.join(FONT_DIR, "Inter-Regular.otf"), 25)
    rgb, a = text_layer([
        ("Oceano Águas Lindas", f_b, cx, 290, (255, 255, 255, 255)),
        ("Qd 25 Lote 1 - Jardim Brasília", f_r, cx, 334, (236, 228, 216, 255)),
        ("Águas Lindas - GO", f_r, cx, 368, (236, 228, 216, 255)),
    ])
    art = art * (1 - a[..., None]) + rgb * a[..., None]
    # filete dourado
    line = np.zeros((H, W), np.float32)
    lx = np.abs(xx - cx)
    line += (np.abs(yy - 238) < 0.9) * _smooth(170, 40, lx)
    art = art * (1 - line[..., None] * 0.85) + GOLD * 0.85 * line[..., None]

    # ---- vinheta
    vx = (xx - W / 2) / (W / 2)
    vy = (yy - H / 2) / (H / 2)
    vig = 1 - 0.35 * np.clip(vx ** 2 * 0.6 + vy ** 2 * 0.25, 0, 1)
    A.vig = vig[..., None].astype(np.float32)

    A.art = art.astype(np.float32)
    A.art_img = to_img(A.art)

    # ---- feixes de luz procedurais (geometria pre-calculada)
    A.beams = []
    for ox, oy, ang0, width, strength in [
        (A.baby[0] + 40, -520.0, -0.03, 0.16, 0.20),  # luz sobre o presepio
        (1070.0, -700.0, 0.0, 0.11, 0.13),             # reforco do feixe do titulo
    ]:
        th_ = np.arctan2(xx - ox, yy - oy).astype(np.float32)
        d = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2).astype(np.float32)
        cone = np.exp(-((th_ - ang0) / width) ** 2).astype(np.float32)
        fall = (0.55 + 0.45 * _smooth(H + 600 + oy * 0.2, -oy, d)).astype(np.float32)
        A.beams.append((th_, cone * fall * strength))

    # ---- poeira dourada (loop perfeito)
    rng = np.random.default_rng(3)
    n = 230
    grp = rng.random(n)
    px = np.where(grp < 0.42, rng.normal(A.baby[0] + 30, 150, n),
                  np.where(grp < 0.75, rng.normal(1075, 110, n), rng.uniform(0, W, n)))
    A.dust = dict(
        x=px.astype(np.float32),
        y=rng.uniform(-20, H + 20, n).astype(np.float32),
        phase=rng.random(n).astype(np.float32),
        life_div=rng.choice([1, 2], n),            # vida = LOOP_T / div
        vy=rng.uniform(-14, -4, n).astype(np.float32),
        sway=rng.uniform(4, 16, n).astype(np.float32),
        sway_k=rng.choice([1, 2, 3], n),
        size=rng.uniform(0.7, 2.2, n).astype(np.float32),
        bri=(rng.uniform(0.25, 0.9, n) * np.where(grp < 0.75, 1.0, 0.45)).astype(np.float32),
    )
    return A


def lum_full(a):
    return a @ np.array([0.30, 0.59, 0.11], np.float32)


# ---------------------------------------------------------------------------
# Primitivas de luz
# ---------------------------------------------------------------------------
def add_gauss(img, x, y, sigma, color, amp):
    r = int(sigma * 3.5) + 2
    x0, x1 = int(x) - r, int(x) + r + 1
    y0, y1 = int(y) - r, int(y) + r + 1
    X0, X1, Y0, Y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    gx = np.exp(-0.5 * ((np.arange(X0, X1, dtype=np.float32) - x) / sigma) ** 2)
    gy = np.exp(-0.5 * ((np.arange(Y0, Y1, dtype=np.float32) - y) / sigma) ** 2)
    img[Y0:Y1, X0:X1] += (gy[:, None] * gx[None, :] * amp)[..., None] * color


def add_star(img, x, y, core, glow, spike_len, spike_amp, rot=0.0, color=GOLD, win=None):
    """Estrela: nucleo + halo + 4 raios principais + 4 raios diagonais menores."""
    R = int(win or max(spike_len * 1.1, glow * 4.5, 40))
    X0, X1 = max(int(x) - R, 0), min(int(x) + R + 1, W)
    Y0, Y1 = max(int(y) - R, 0), min(int(y) + R + 1, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    gx = np.arange(X0, X1, dtype=np.float32) - x
    gy = np.arange(Y0, Y1, dtype=np.float32) - y
    dx, dy = np.meshgrid(gx, gy)
    r2 = dx * dx + dy * dy
    e = np.zeros_like(dx)
    e += 1.6 * np.exp(-r2 / (2 * core * core))
    e += 0.55 * np.exp(-np.sqrt(r2) / glow)
    out = WARM_WHITE * e[..., None] * 0 + 0
    col_core = WARM_WHITE
    acc = e[..., None] * col_core
    if spike_amp > 0:
        s = np.zeros_like(dx)
        for k in range(4):
            a = rot + k * math.pi / 2
            u = dx * math.cos(a) + dy * math.sin(a)
            v = -dx * math.sin(a) + dy * math.cos(a)
            s += np.exp(-(v / (1.1 + 0.004 * np.abs(u))) ** 2) * np.exp(-np.maximum(u, 0) / (spike_len * 0.33)) * (u > 0)
        for k in range(4):
            a = rot + math.pi / 4 + k * math.pi / 2
            u = dx * math.cos(a) + dy * math.sin(a)
            v = -dx * math.sin(a) + dy * math.cos(a)
            s += 0.45 * np.exp(-(v / 0.9) ** 2) * np.exp(-np.maximum(u, 0) / (spike_len * 0.12)) * (u > 0)
        acc += (s * spike_amp)[..., None] * color
    img[Y0:Y1, X0:X1] += acc


def dust_layer(A, t, img):
    d = A.dust
    n = len(d["x"])
    for i in range(n):
        life = LOOP_T / d["life_div"][i]
        age = ((t / life) + d["phase"][i]) % 1.0
        a = math.sin(math.pi * age) ** 2 * d["bri"][i]
        if a < 0.02:
            continue
        x = d["x"][i] + d["sway"][i] * math.sin(2 * math.pi * (d["sway_k"][i] * t / LOOP_T + d["phase"][i]))
        y = d["y"][i] + d["vy"][i] * (age - 0.5) * life
        tw = 0.75 + 0.25 * math.sin(2 * math.pi * (7 * t / LOOP_T + d["phase"][i] * 5))
        add_gauss(img, x, y, d["size"][i], GOLD, a * tw * 0.9)


# ---------------------------------------------------------------------------
# Arte animada (loop perfeito de LOOP_T segundos)
# ---------------------------------------------------------------------------
def art_frame(A, t):
    t = t % LOOP_T
    ph = 2 * math.pi * t / LOOP_T

    # respiracao suave da camera (zoom lento ida e volta)
    z = 1.006 + 0.012 * (0.5 - 0.5 * math.cos(ph))  # nunca 1.0: nitidez constante
    if True:  # sempre reamostra, para o quadro 0 ficar identico aos demais
        im = A.art_img
        cx, cy = 960, 240
        im = im.transform((W, H), Image.AFFINE,
                          (1 / z, 0, cx - cx / z, 0, 1 / z, cy - cy / z),
                          resample=Image.BICUBIC)
        f = np.asarray(im).astype(np.float32) / 255.0
    else:
        f = A.art.copy()

    # luz do bebe pulsando
    glow = 0.10 + 0.05 * math.sin(ph * 2)
    bx, by = A.baby
    g = np.exp(-(((A.xx - bx) / 170) ** 2 + ((A.yy - by) / 110) ** 2))
    f += g[..., None] * GOLD * glow

    # feixes de luz com raios que se movem
    rays = np.zeros((H, W), np.float32)
    for k, (th, w) in enumerate(A.beams):
        s = (0.55
             + 0.20 * np.sin(th * 61 + ph * 1 + k)
             + 0.15 * np.sin(th * 137 - ph * 2 + 2 * k)
             + 0.10 * np.sin(th * 23 + ph * 3))
        rays += w * s * (0.85 + 0.15 * math.sin(ph + k * 2))
    light = rays[..., None] * np.array([1.0, 0.78, 0.50], np.float32)
    f = 1 - (1 - f) * (1 - np.clip(light, 0, 1))

    # brilho que atravessa o titulo (uma passada por loop)
    sweep_t = (t - 1.0) / 2.4
    if 0 <= sweep_t <= 1:
        pos = -300 + sweep_t * (1000 + 600)
        band = np.exp(-((A.xx - A.title_x0 - pos - (A.yy - 256) * 0.45) / 70) ** 2)
        f += (band * A.letters)[..., None] * np.array([0.55, 0.45, 0.30], np.float32)

    # cintilacoes nas letras
    for i, (gx, gy) in enumerate(A.glints):
        c = (t - (3.6 + i * 1.05)) / 0.9
        if 0 < c < 1:
            k = math.sin(math.pi * c) ** 2
            add_star(f, gx, gy, core=1.6, glow=6, spike_len=55 * k + 5, spike_amp=0.9 * k,
                     rot=0.3 * c, color=WARM_WHITE)

    dust_layer(A, t, f)
    f *= A.vig
    return f


# ---------------------------------------------------------------------------
# Introducao: estrela sobe sobre fundo chroma, brilha, expande e revela a arte
# ---------------------------------------------------------------------------
def star_pos(s):
    u = ease_in_out(s / RISE_END)
    u = 1 - (1 - u) ** 1.6  # chega devagar no topo
    x = STAR_START[0] + (STAR_END[0] - STAR_START[0]) * u + 22 * math.sin(u * math.pi * 1.5) * (1 - u)
    y = STAR_START[1] + (STAR_END[1] - STAR_START[1]) * u
    return x, y


def _trail_particles():
    rng = np.random.default_rng(11)
    parts = []
    dt = 1.0 / FPS
    s = 0.15
    while s < RISE_END + 0.3:
        x, y = star_pos(s)
        for _ in range(3):
            parts.append(dict(t0=s, x=x + rng.normal(0, 3), y=y + rng.normal(0, 3),
                              vx=rng.normal(0, 14), vy=rng.uniform(10, 45),
                              life=rng.uniform(0.6, 1.3), size=rng.uniform(0.8, 2.0),
                              bri=rng.uniform(0.4, 1.0)))
        s += dt
    # explosao de faiscas na expansao
    for _ in range(160):
        a = rng.uniform(0, 2 * math.pi)
        sp = rng.uniform(150, 900)
        parts.append(dict(t0=REVEAL_START + rng.uniform(0, 0.15), x=STAR_END[0], y=STAR_END[1],
                          vx=math.cos(a) * sp, vy=math.sin(a) * sp * 0.8,
                          life=rng.uniform(0.6, 1.6), size=rng.uniform(1.0, 2.6),
                          bri=rng.uniform(0.5, 1.0), drag=True))
    return parts


TRAIL = _trail_particles()


def intro_frame(A, s, alpha_mode=False):
    """Retorna (rgb, alpha). Com alpha_mode=False o fundo e verde chroma."""
    E = np.zeros((H, W, 3), np.float32)   # emissao de luz (estrela, faiscas...)

    x, y = star_pos(min(s, RISE_END))
    appear = smooth(0.0, 0.5, s)

    # ---- estrela
    if s < REVEAL_START + 1.2:
        if s < RISE_END:
            tw = 0.85 + 0.15 * math.sin(s * 22)
            add_star(E, x, y, core=2.4 + 1.0 * appear, glow=7 + 3 * appear,
                     spike_len=40 + 25 * (s / RISE_END), spike_amp=0.55 * appear * tw,
                     rot=0.15 * math.sin(s * 2.0), win=140)
            E *= appear
        elif s < REVEAL_START:
            c = (s - RISE_END) / (SHINE_END - RISE_END)
            pulse = math.sin(math.pi * c)
            tw = 1 + 0.35 * math.sin(c * math.pi * 4) ** 2
            add_star(E, x, y, core=3.4 + 2.5 * pulse, glow=10 + 16 * pulse,
                     spike_len=65 + 260 * ease_out_cubic(c), spike_amp=(0.6 + 0.9 * pulse) * tw,
                     rot=0.15 * math.sin(s * 2.0) + 0.35 * c, win=420)
        else:
            c = (s - REVEAL_START) / 1.2
            k = 1 - smooth(0, 1, c)
            add_star(E, x, y, core=5 + 8 * c, glow=26 + 40 * c,
                     spike_len=330 + 500 * c, spike_amp=1.6 * k,
                     rot=0.15 * math.sin(s * 2.0) + 0.35 + 0.2 * c, win=900)
            E *= k

    # ---- particulas (rastro + faiscas)
    for p in TRAIL:
        age = s - p["t0"]
        if age < 0 or age > p["life"]:
            continue
        u = age / p["life"]
        if p.get("drag"):
            dist = (1 - math.exp(-3 * age))/3
            px, py = p["x"] + p["vx"] * dist, p["y"] + p["vy"] * dist + 40 * age * age
        else:
            px, py = p["x"] + p["vx"] * age, p["y"] + p["vy"] * age
        a = p["bri"] * (1 - u) ** 1.5 * (0.7 + 0.3 * math.sin(age * 40 + p["t0"] * 13))
        add_gauss(E, px, py, p["size"], GOLD, a)

    # ---- revelacao (iris circular a partir da estrela)
    sx, sy = STAR_END
    dist = np.sqrt((A.xx - sx) ** 2 + (A.yy - sy) ** 2)
    maxr = 1150.0
    if s >= REVEAL_START:
        c = (s - REVEAL_START) / (REVEAL_END - REVEAL_START)
        r = maxr * ease_in_cubic(min(c, 1.0)) ** 0.55 if c < 1 else maxr + 200
        feather = 35 + 70 * min(c, 1)
        M = 1 - _smooth(r - feather, r, dist)
        # anel de luz na borda
        ring = np.exp(-((dist - r + feather * 0.4) / (25 + 30 * min(c, 1))) ** 2) * (1 - smooth(0.75, 1.0, c))
        E += ring[..., None] * GOLD * 0.9
        # clarao
        fl = math.exp(-((s - REVEAL_START - 0.15) / 0.35) ** 2)
        E += (np.exp(-dist / 190) * fl * 1.4)[..., None] * WARM_WHITE
        artf = art_frame(A, s - REVEAL_START)
        # arte "superexposta" no inicio, assentando
        expo = 0.6 * math.exp(-(s - REVEAL_START) / 0.45) * (1 - smooth(REVEAL_START + 1.2, INTRO_LEN, s))
        artf = artf + expo * (1 - artf) * np.exp(-dist / 700)[..., None]
    else:
        M = np.zeros((H, W), np.float32)
        artf = None

    M3 = M[..., None]
    if artf is not None:
        inside = 1 - (1 - np.clip(artf, 0, 1)) * (1 - np.clip(E, 0, 1))
    else:
        inside = np.zeros((H, W, 3), np.float32)

    # fora da iris: emissao sobre fundo chroma / transparente
    Ec = np.clip(E, 0, 1)
    ea = np.clip(Ec.max(axis=2), 0, 1)
    if alpha_mode:
        prem = inside * M3 + Ec * (1 - M3)
        alpha = M + ea * (1 - M)
        rgb = prem / np.maximum(alpha[..., None], 1e-4)
        return np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)
    outside = CHROMA * (1 - ea[..., None]) + Ec
    rgb = inside * M3 + outside * (1 - M3)
    return np.clip(rgb, 0, 1), None


def to_u8(a, seed=0):
    # leve dither para evitar banding no escuro
    n = np.random.default_rng(seed).random(a.shape[:2], dtype=np.float32)[..., None] - 0.5
    return np.clip(a * 255 + n + 0.5, 0, 255).astype(np.uint8)
