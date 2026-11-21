"""
O Autor da Historia - motion graphic para o telao (1920x512).

  build_assets()        -> separa a arte do flyer em camadas (fundo, textos, logo...)
  art_frame(A, t, s)    -> arte animada; t = tempo do loop (periodo LOOP_T),
                           s = tempo absoluto do video (controla as entradas dos
                           textos). s=None -> tudo ja entrou (loop puro).
  intro_frame(A, s)     -> estrela cartunesca sobre fundo chroma: sobe, brilha,
                           cresce em forma de estrela e revela a arte.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 512
FPS = 30
LOOP_T = 10.0              # periodo do loop da arte (s)

# ---- linha do tempo da introducao (s)
RISE_END = 1.25            # estrela chega ao topo (subida rapida)
SHINE_START = 1.55         # depois de "quicar" no topo, brilha
REVEAL_START = 2.45        # clarao de luz sai da estrela
FLASH_FULL = REVEAL_START + 0.35   # tela toda clara
FLASH_FADE = FLASH_FULL + 0.12     # comeca a clarear revelando a arte
REVEAL_END = FLASH_FADE + 0.90
INTRO_LEN = 4.6            # depois disso nao ha mais faiscas da intro

# ---- entradas dos textos (inicio, duracao) em tempo absoluto do video
ENTR = {
    "sub":   (3.35, 0.70),   # "Especial de Natal"
    "title": (3.60, 1.50),   # "O Autor da Historia" (escrito por um traco de luz)
    "date":  (4.80, 0.70),   # "20 de dezembro | 19h"
    "logo":  (5.05, 0.80),
    "line":  (5.40, 0.65),
    "t1":    (5.55, 0.65),
    "t2":    (5.70, 0.65),
    "t3":    (5.85, 0.65),
}
ENTR_DONE = 6.65

CHROMA = np.array([0.0, 1.0, 0.0], np.float32)  # verde puro #00FF00
GOLD = np.array([1.0, 0.80, 0.52], np.float32)
WARM_WHITE = np.array([1.0, 0.95, 0.85], np.float32)

STAR_START = (960.0, 470.0)
STAR_END = (1062.0, 84.0)    # topo do feixe de luz da arte
STAR_R = 34.0                # raio da estrela cartunesca (px)

# cores da estrela cartunesca
C_OUT = (196, 112, 8)
C_BODY = (255, 206, 40)
C_LIGHT = (255, 238, 140)
C_SPARK = (255, 232, 120)

FONT_DIR = "/usr/share/fonts/opentype/inter"


# ---------------------------------------------------------------------------
# utilitarios
# ---------------------------------------------------------------------------
def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(t))


def ease_out_cubic(t):
    return 1 - (1 - clamp01(t)) ** 3


def ease_out_back(t, k=1.70158):
    t = clamp01(t) - 1
    return 1 + (k + 1) * t ** 3 + k * t ** 2


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def resize(a, w, h):
    return np.asarray(to_img(a).resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0


def lum(a):
    return a @ np.array([0.30, 0.59, 0.11], np.float32)


def box_blur(a, r, passes=3):
    """Desfoque separavel (aprox. gaussiano) em float."""
    for _ in range(passes):
        for ax in (0, 1):
            p = np.concatenate([np.repeat(a.take([0], ax), r + 1, ax), a,
                                np.repeat(a.take([-1], ax), r, ax)], ax)
            c = np.cumsum(p, axis=ax, dtype=np.float64)
            n = a.shape[ax]
            hi = c.take(range(2 * r + 1, 2 * r + 1 + n), ax)
            lo = c.take(range(0, n), ax)
            a = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return a


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
    h, w = src.shape[:2]
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if dx1 <= dx0 or dy1 <= dy0:
        return
    s = src[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    m = mask[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0, None]
    dst[dy0:dy1, dx0:dx1] = dst[dy0:dy1, dx0:dx1] * (1 - m) + s * m


def shift_v(a, dy):
    """Desloca verticalmente com subpixel (preenche com zero)."""
    if abs(dy) < 1e-4:
        return a
    i = math.floor(dy)
    fr = dy - i

    def sh(arr, k):
        out = np.zeros_like(arr)
        if k >= arr.shape[0] or -k >= arr.shape[0]:
            return out
        if k >= 0:
            out[k:] = arr[:arr.shape[0] - k]
        else:
            out[:k] = arr[-k:]
        return out
    return sh(a, i) * (1 - fr) + sh(a, i + 1) * fr


class Sprite:
    """Elemento RGBA (textos, logo) posicionado pelo centro."""

    def __init__(self, img, cx, cy):
        self.img = img            # PIL RGBA
        self.cx, self.cy = cx, cy

    def draw(self, f, dx=0.0, dy=0.0, scale=1.0, opacity=1.0):
        if opacity <= 0.002:
            return
        w, h = self.img.size
        pad = 24
        ow, oh = w + 2 * pad, h + 2 * pad
        cx, cy = self.cx + dx, self.cy + dy
        X0, Y0 = int(math.floor(cx - ow / 2)), int(math.floor(cy - oh / 2))
        lx, ly = cx - X0, cy - Y0
        inv = 1.0 / scale
        im = self.img.transform((ow, oh), Image.AFFINE,
                                (inv, 0, w / 2 - lx * inv, 0, inv, h / 2 - ly * inv),
                                resample=Image.BICUBIC)
        a = np.asarray(im).astype(np.float32) / 255.0
        paste(f, a[..., :3], a[..., 3] * opacity, X0, Y0)


def text_sprite(txt, font, color, cx, cy):
    d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    l, t, r, b = d.textbbox((0, 0), txt, font=font, anchor="mm")
    w, h = r - l + 8, b - t + 8
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((w / 2, h / 2), txt, font=font, fill=color, anchor="mm")
    return Sprite(img, cx, cy)


# ---------------------------------------------------------------------------
# Montagem das camadas
# ---------------------------------------------------------------------------
class Assets:
    pass


def build_assets():
    A = Assets()
    flyer = load(os.path.join(HERE, "flyer.jpg"))
    ref = load(os.path.join(HERE, "telao_ref.jpg"))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A.yy, A.xx = yy, xx

    # ---- fundo escuro quente
    bg = np.zeros((H, W, 3), np.float32)
    bg[:] = np.array([0.020, 0.013, 0.007], np.float32)
    halo = np.exp(-(((xx - 1000) / 900) ** 2 + ((yy - 200) / 420) ** 2))
    bg += halo[..., None] * np.array([0.045, 0.028, 0.012], np.float32)
    halo2 = np.exp(-(((xx - 300) / 420) ** 2 + ((yy - 330) / 300) ** 2))
    bg += halo2[..., None] * np.array([0.05, 0.030, 0.012], np.float32)
    base = bg.copy()

    # ---- presepio (esquerda)
    nat = flyer[780:1400, 80:900]
    nh = H + 8
    nw = int(round(nat.shape[1] * nh / nat.shape[0]))
    nat = resize(nat, nw, nh)
    nm = feather_mask(nh, nw, left=70, right=150, top=40, bottom=0)
    A.nat_x0 = -40
    paste(base, nat, nm, A.nat_x0, -4)
    sc = nh / 620.0
    A.baby = (A.nat_x0 + (430 - 80) * sc, -4 + (1145 - 780) * sc)

    # ---- coluna do titulo: separa os textos do feixe de luz
    ttl = flyer[240:770, 10:890]
    th = H
    tw = int(round(ttl.shape[1] * th / ttl.shape[0]))
    ttl = resize(ttl, tw, th)
    A.tw = tw
    A.title_x0 = 960 - tw // 2
    tmask = feather_mask(th, tw, left=40, right=40, top=0, bottom=0)

    ty = np.arange(th, dtype=np.float32)[:, None]
    bands = {  # faixas verticais (px da coluna) de cada texto
        "sub": (20, 74),
        "title": (74, 428),
        "date": (430, 506),
    }
    band_all = np.zeros((th, 1), np.float32)
    band_m = {}
    for k, (a0, a1) in bands.items():
        m = _smooth(a0 - 4, a0 + 2, ty) * (1 - _smooth(a1 - 2, a1 + 4, ty))
        band_m[k] = m
        band_all = np.maximum(band_all, m)

    L = lum(ttl)
    # fundo sem textos: interpola verticalmente o feixe de luz entre as linhas
    # limpas acima e abaixo dos textos (o feixe e praticamente vertical)
    r0, r1 = 16, 502
    top = box_blur(ttl[4:14].mean(0, keepdims=True), 4, 2)
    bot = box_blur(ttl[503:511].mean(0, keepdims=True), 4, 2)
    v = np.clip((ty - 9) / (507 - 9), 0, 1)[..., None]
    interp = top * (1 - v) + bot * v
    fill = (_smooth(r0 - 6, r0, ty) * (1 - _smooth(r1, r1 + 6, ty)))[..., None]
    plate = ttl * (1 - fill) + interp * fill
    D = (ttl - plate) * tmask[..., None]
    A.D = {k: (D * band_m[k][..., None]).astype(np.float32) for k in bands}
    A.plate_col = plate
    paste(base, plate, tmask, A.title_x0, 0)

    # mascara das letras douradas
    warm = ttl[..., 0] - ttl[..., 2]
    letters = _smooth(0.42, 0.62, L) * _smooth(0.10, 0.22, warm) * band_m["title"]
    A.letters_col = (letters * tmask).astype(np.float32)
    Lf = np.zeros((H, W), np.float32)
    Lf[:, A.title_x0:A.title_x0 + tw] = A.letters_col
    A.letters = Lf
    A.letter_pts = np.argwhere(A.letters_col > 0.8).astype(np.float32)

    # ---- coluna da direita
    cx = 1652
    logo = ref[418:490, 840:1080]
    la = _smooth(0.30, 0.85, logo[..., 0])
    lh, lw = la.shape
    s = 1.55
    la = np.asarray(Image.fromarray((la * 255).astype(np.uint8)).resize(
        (int(lw * s), int(lh * s)), Image.LANCZOS)).astype(np.float32) / 255.0
    lh, lw = la.shape
    grad = np.linspace(0, 1, lh, dtype=np.float32)[:, None, None]
    lrgb = (np.array([0.93, 0.80, 0.58], np.float32) * (1 - grad)
            + np.array([0.72, 0.53, 0.30], np.float32) * grad)
    lrgb = np.broadcast_to(lrgb, (lh, lw, 3))
    rgba = np.concatenate([lrgb, la[..., None]], axis=2)
    A.logo = Sprite(Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA"), cx, 118 + lh / 2)

    f_b = ImageFont.truetype(os.path.join(FONT_DIR, "Inter-Bold.otf"), 34)
    f_r = ImageFont.truetype(os.path.join(FONT_DIR, "Inter-Regular.otf"), 25)
    A.t1 = text_sprite("Oceano Águas Lindas", f_b, (255, 255, 255, 255), cx, 290)
    A.t2 = text_sprite("Qd 25 Lote 1 - Jardim Brasília", f_r, (236, 228, 216, 255), cx, 334)
    A.t3 = text_sprite("Águas Lindas - GO", f_r, (236, 228, 216, 255), cx, 368)
    A.line_cx, A.line_y = cx, 238

    A.base = base.astype(np.float32)

    vx = (xx - W / 2) / (W / 2)
    vy = (yy - H / 2) / (H / 2)
    A.vig = (1 - 0.35 * np.clip(vx ** 2 * 0.6 + vy ** 2 * 0.25, 0, 1))[..., None].astype(np.float32)

    # ---- arte final (tudo ja entrou)
    A.art = compose(A, None)
    A.art_img = to_img(A.art)

    cand = np.argwhere((Lf > 0.9) & (lum(A.art) > 0.80))
    rng = np.random.default_rng(7)
    rng.shuffle(cand)
    picks = []
    for y, x in cand:
        if all((x - px) ** 2 + (y - py) ** 2 > 140 ** 2 for py, px in picks):
            picks.append((y, x))
        if len(picks) >= 6:
            break
    A.glints = [(float(x), float(y)) for y, x in picks]

    # ---- feixes de luz procedurais
    A.beams = []
    for ox, oy, ang0, width, strength in [
        (A.baby[0] + 40, -520.0, -0.03, 0.16, 0.20),
        (1070.0, -700.0, 0.0, 0.11, 0.13),
    ]:
        th_ = np.arctan2(xx - ox, yy - oy).astype(np.float32)
        d = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2).astype(np.float32)
        cone = np.exp(-((th_ - ang0) / width) ** 2).astype(np.float32)
        fall = (0.55 + 0.45 * _smooth(H + 600 + oy * 0.2, -oy, d)).astype(np.float32)
        A.beams.append((th_, cone * fall * strength))

    # ---- poeira dourada
    rng = np.random.default_rng(3)
    n = 230
    grp = rng.random(n)
    px = np.where(grp < 0.42, rng.normal(A.baby[0] + 30, 150, n),
                  np.where(grp < 0.75, rng.normal(1075, 110, n), rng.uniform(0, W, n)))
    A.dust = dict(
        x=px.astype(np.float32),
        y=rng.uniform(-20, H + 20, n).astype(np.float32),
        phase=rng.random(n).astype(np.float32),
        life_div=rng.choice([1, 2], n),
        vy=rng.uniform(-14, -4, n).astype(np.float32),
        sway=rng.uniform(4, 16, n).astype(np.float32),
        sway_k=rng.choice([1, 2, 3], n),
        size=rng.uniform(0.7, 2.2, n).astype(np.float32),
        bri=(rng.uniform(0.25, 0.9, n) * np.where(grp < 0.75, 1.0, 0.45)).astype(np.float32),
    )
    return A


def entr(s, key):
    if s is None:
        return 1.0
    t0, d = ENTR[key]
    return clamp01((s - t0) / d)


def add_col(f, A, layer):
    f[:, A.title_x0:A.title_x0 + A.tw] += layer


def compose(A, s):
    """Monta a arte com as entradas dos textos no tempo absoluto s."""
    f = A.base.copy()

    # "Especial de Natal": desce com um quiquezinho
    p = entr(s, "sub")
    if p > 0:
        e = ease_out_back(p, 2.2)
        add_col(f, A, shift_v(A.D["sub"], -26 * (1 - e)) * smooth(0, 0.55, p))

    # titulo: "escrito" por um traco de luz diagonal da esquerda para a direita
    p = entr(s, "title")
    if p > 0:
        tw = A.tw
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(tw, dtype=np.float32)[None, :]
        u = xs + (ys - 250) * 0.35
        e = -260 + ease_in_out(p) * (tw + 520)
        rev = _smooth(e + 10, e - 70, u)
        layer = A.D["title"] * rev[..., None]
        k = 1 - smooth(0.85, 1.0, p)
        if k > 0:
            edge = np.exp(-((u - e + 22) / 26) ** 2) * A.letters_col * k
            layer = layer + edge[..., None] * np.array([1.0, 0.85, 0.55], np.float32) * 1.4
            halo = np.exp(-((u - e) / 120) ** 2) * k * 0.10
            layer = layer + halo[..., None] * GOLD * (A.letters_col[..., None] * 0.6 + 0.15)
        add_col(f, A, layer)
        if k > 0:
            rng = np.random.default_rng(int(p * 10000))
            pts = A.letter_pts
            uu = pts[:, 1] + (pts[:, 0] - 250) * 0.35
            near = pts[np.abs(uu - e + 15) < 14]
            if len(near):
                sel = near[rng.integers(0, len(near), min(10, len(near)))]
                for (py, px) in sel:
                    add_star(f, A.title_x0 + px + rng.normal(0, 4), py + rng.normal(0, 4),
                             core=1.2, glow=3, spike_len=rng.uniform(8, 22),
                             spike_amp=0.7 * k, rot=rng.uniform(0, 1), color=WARM_WHITE, win=30)

    # data: sobe e aparece
    p = entr(s, "date")
    if p > 0:
        e = ease_out_back(p, 1.6)
        add_col(f, A, shift_v(A.D["date"], 30 * (1 - e)) * smooth(0, 0.55, p))

    # logo: cresce com quique
    p = entr(s, "logo")
    if p > 0:
        A.logo.draw(f, scale=0.6 + 0.4 * ease_out_back(p, 2.4), opacity=smooth(0, 0.45, p))

    # filete dourado: abre do centro para as pontas
    p = entr(s, "line")
    if p > 0:
        half = 170 * ease_out_cubic(p)
        lx = np.abs(np.arange(W, dtype=np.float32) - A.line_cx)
        prof = _smooth(half, max(half - 130, 0.0) - 1e-3, lx) if half > 0.5 else np.zeros(W, np.float32)
        y0 = A.line_y
        m = prof[None, :, None]
        f[y0:y0 + 1] = f[y0:y0 + 1] * (1 - m * 0.85) + GOLD * 0.85 * m
        k = 1 - smooth(0.7, 1.0, p)
        if k > 0:
            for sgn in (-1, 1):
                add_gauss(f, A.line_cx + sgn * half, y0, 3.5, GOLD, 1.2 * k)

    # textos da direita: sobem em sequencia
    for key, spr in (("t1", A.t1), ("t2", A.t2), ("t3", A.t3)):
        p = entr(s, key)
        if p > 0:
            spr.draw(f, dy=22 * (1 - ease_out_back(p, 1.4)), opacity=smooth(0, 0.6, p))
    return f


# ---------------------------------------------------------------------------
# Primitivas de luz
# ---------------------------------------------------------------------------
def add_gauss(img, x, y, sigma, color, amp):
    r = int(sigma * 3.5) + 2
    X0, X1 = max(int(x) - r, 0), min(int(x) + r + 1, img.shape[1])
    Y0, Y1 = max(int(y) - r, 0), min(int(y) + r + 1, img.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    gx = np.exp(-0.5 * ((np.arange(X0, X1, dtype=np.float32) - x) / sigma) ** 2)
    gy = np.exp(-0.5 * ((np.arange(Y0, Y1, dtype=np.float32) - y) / sigma) ** 2)
    img[Y0:Y1, X0:X1] += (gy[:, None] * gx[None, :] * amp)[..., None] * color


def add_star(img, x, y, core, glow, spike_len, spike_amp, rot=0.0, color=GOLD, win=None):
    """Cintilacao realista (nucleo + halo + raios)."""
    R = int(win or max(spike_len * 1.1, glow * 4.5, 40))
    X0, X1 = max(int(x) - R, 0), min(int(x) + R + 1, img.shape[1])
    Y0, Y1 = max(int(y) - R, 0), min(int(y) + R + 1, img.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    gx = np.arange(X0, X1, dtype=np.float32) - x
    gy = np.arange(Y0, Y1, dtype=np.float32) - y
    dx, dy = np.meshgrid(gx, gy)
    r2 = dx * dx + dy * dy
    e = 1.6 * np.exp(-r2 / (2 * core * core)) + 0.55 * np.exp(-np.sqrt(r2) / glow)
    acc = e[..., None] * WARM_WHITE
    if spike_amp > 0:
        s = np.zeros_like(dx)
        for k in range(4):
            a = rot + k * math.pi / 2
            u = dx * math.cos(a) + dy * math.sin(a)
            v = -dx * math.sin(a) + dy * math.cos(a)
            s += np.exp(-(v / (1.1 + 0.004 * np.abs(u))) ** 2) * np.exp(-np.maximum(u, 0) / (spike_len * 0.33)) * (u > 0)
        acc += (s * spike_amp)[..., None] * color
    img[Y0:Y1, X0:X1] += acc


def dust_layer(A, t, img, amt=1.0):
    d = A.dust
    for i in range(len(d["x"])):
        life = LOOP_T / d["life_div"][i]
        age = ((t / life) + d["phase"][i]) % 1.0
        a = math.sin(math.pi * age) ** 2 * d["bri"][i] * amt
        if a < 0.02:
            continue
        x = d["x"][i] + d["sway"][i] * math.sin(2 * math.pi * (d["sway_k"][i] * t / LOOP_T + d["phase"][i]))
        y = d["y"][i] + d["vy"][i] * (age - 0.5) * life
        tw = 0.75 + 0.25 * math.sin(2 * math.pi * (7 * t / LOOP_T + d["phase"][i] * 5))
        add_gauss(img, x, y, d["size"][i], GOLD, a * tw * 0.9)


# ---------------------------------------------------------------------------
# Arte animada
# ---------------------------------------------------------------------------
def art_frame(A, t, s=None):
    t = t % LOOP_T
    ph = 2 * math.pi * t / LOOP_T
    entering = s is not None and s < ENTR_DONE
    amb = 1.0 if s is None else 0.25 + 0.75 * smooth(REVEAL_START, REVEAL_END + 1.0, s)
    deco = 1.0 if s is None else smooth(ENTR_DONE, ENTR_DONE + 0.5, s)

    im = to_img(compose(A, s)) if entering else A.art_img
    z = 1.006 + 0.012 * (0.5 - 0.5 * math.cos(ph))   # nunca 1.0: nitidez constante
    cx, cy = 960, 240
    im = im.transform((W, H), Image.AFFINE, (1 / z, 0, cx - cx / z, 0, 1 / z, cy - cy / z),
                      resample=Image.BICUBIC)
    f = np.asarray(im).astype(np.float32) / 255.0

    glow = (0.10 + 0.05 * math.sin(ph * 2)) * amb
    bx, by = A.baby
    g = np.exp(-(((A.xx - bx) / 170) ** 2 + ((A.yy - by) / 110) ** 2))
    f += g[..., None] * GOLD * glow

    rays = np.zeros((H, W), np.float32)
    for k, (th, w) in enumerate(A.beams):
        sgl = (0.55 + 0.20 * np.sin(th * 61 + ph + k) + 0.15 * np.sin(th * 137 - ph * 2 + 2 * k)
               + 0.10 * np.sin(th * 23 + ph * 3))
        rays += w * sgl * (0.85 + 0.15 * math.sin(ph + k * 2))
    light = rays[..., None] * np.array([1.0, 0.78, 0.50], np.float32) * amb
    f = 1 - (1 - f) * (1 - np.clip(light, 0, 1))

    if deco > 0:
        sweep_t = (t - 1.0) / 2.4
        if 0 <= sweep_t <= 1:
            pos = -300 + sweep_t * 1600
            band = np.exp(-((A.xx - A.title_x0 - pos - (A.yy - 256) * 0.45) / 70) ** 2)
            f += (band * A.letters * deco)[..., None] * np.array([0.55, 0.45, 0.30], np.float32)
        for i, (gx, gy) in enumerate(A.glints):
            c = (t - (3.6 + i * 1.05)) / 0.9
            if 0 < c < 1:
                k = math.sin(math.pi * c) ** 2 * deco
                add_star(f, gx, gy, core=1.6, glow=6, spike_len=55 * k + 5, spike_amp=0.9 * k,
                         rot=0.3 * c, color=WARM_WHITE)

    dust_layer(A, t, f, amb)
    f *= A.vig
    return f


# ---------------------------------------------------------------------------
# Introducao: estrela cartunesca
# ---------------------------------------------------------------------------
SS = 2  # supersampling dos desenhos vetoriais


def star_points(cx, cy, R, rot, k=0.5, sx=1.0, sy=1.0):
    pts = []
    for i in range(10):
        a = rot - math.pi / 2 + i * math.pi / 5
        r = R if i % 2 == 0 else R * k
        pts.append((cx + r * math.cos(a) * sx, cy + r * math.sin(a) * sy))
    return pts


def _rounded_star(d, pts, fill, lw):
    P = [(x * SS, y * SS) for x, y in pts]
    d.polygon(P, fill=fill)
    if lw > 0:
        d.line(P + [P[0], P[1]], fill=fill, width=max(int(lw * SS), 1), joint="curve")
        r = lw * SS / 2
        for x, y in P[::2]:
            d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def draw_cartoon_star(d, cx, cy, R, rot, sx=1.0, sy=1.0, alpha=255):
    a = int(alpha)
    ow = max(R * 0.30, 2.5)
    _rounded_star(d, star_points(cx, cy, R, rot, 0.5, sx, sy), C_OUT + (a,), ow)
    _rounded_star(d, star_points(cx, cy, R * 0.97, rot, 0.5, sx, sy), C_BODY + (a,), ow * 0.35)
    hx, hy = cx - R * 0.10 * sx, cy - R * 0.12 * sy
    _rounded_star(d, star_points(hx, hy, R * 0.52, rot, 0.5, sx, sy), C_LIGHT + (a,), ow * 0.3)
    gx, gy = cx - R * 0.26 * sx, cy - R * 0.30 * sy
    rx, ry = R * 0.13 * sx, R * 0.08 * sy
    d.ellipse(((gx - rx) * SS, (gy - ry) * SS, (gx + rx) * SS, (gy + ry) * SS), fill=(255, 255, 255, a))


def draw_sparkle(d, x, y, r, rot, color, alpha=255):
    """Brilhinho cartunesco de 4 pontas."""
    if r < 0.4:
        return
    pts = []
    for i in range(8):
        a = rot + i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.28
        pts.append(((x + rr * math.cos(a)) * SS, (y + rr * math.sin(a)) * SS))
    d.polygon(pts, fill=tuple(color) + (int(alpha),))


def star_state(s):
    """Posicao, rotacao, escala e esticamento da estrela."""
    u = clamp01(s / RISE_END)
    m = ease_out_back(u, 1.3)                      # passa um pouco do topo e volta
    x = STAR_START[0] + (STAR_END[0] - STAR_START[0]) * m + 26 * math.sin(u * math.pi * 2) * (1 - u)
    y = STAR_START[1] + (STAR_END[1] - STAR_START[1]) * m
    rot = 2 * math.pi * ease_out_cubic(u)          # uma volta completa subindo
    rot += 0.12 * math.sin((s - RISE_END) * 6) * smooth(RISE_END, RISE_END + 0.3, s)  # balancinho
    scale = ease_out_back(clamp01(s / 0.28), 2.5)
    vel = (1 - u) ** 2 if s < RISE_END else 0.0     # estica na subida
    stretch = 0.30 * vel
    if s > RISE_END - 0.15:                         # achata quando para
        q = s - (RISE_END - 0.15)
        stretch += -0.22 * math.exp(-q / 0.16) * math.cos(2 * math.pi * q / 0.32)
    sy = 1 + stretch
    sx = 1 / sy
    return x, y, rot, scale, sx, sy


def _intro_particles():
    rng = np.random.default_rng(11)
    parts = []
    s = 0.08
    while s < RISE_END:
        x, y, *_ = star_state(s)
        for _ in range(2):
            parts.append(dict(t0=s, x=x + rng.normal(0, 8), y=y + 10 + rng.normal(0, 6),
                              vx=rng.normal(0, 30), vy=rng.uniform(20, 80), life=rng.uniform(0.45, 0.9),
                              r=rng.uniform(5, 11), rot=rng.uniform(0, 1), spin=rng.normal(0, 4),
                              kind="sparkle" if rng.random() < 0.6 else "dot", drag=False))
        s += 1.0 / FPS
    for _ in range(70):   # brilhinhos voando enquanto o clarao se dissipa
        a = rng.uniform(0, 2 * math.pi)
        sp = rng.uniform(250, 1100)
        parts.append(dict(t0=FLASH_FADE + rng.uniform(0, 0.12), x=STAR_END[0], y=STAR_END[1],
                          vx=math.cos(a) * sp, vy=math.sin(a) * sp * 0.75, life=rng.uniform(0.7, 1.4),
                          r=rng.uniform(6, 16), rot=rng.uniform(0, 1), spin=rng.normal(0, 5),
                          kind="sparkle" if rng.random() < 0.7 else "dot", drag=True))
    return parts


PARTS = _intro_particles()

# brilhinhos que "pipocam" em volta da estrela quando ela brilha: (dx, dy, inicio, raio)
POPS = [(-118, -6, 0.00, 14), (92, 40, 0.08, 11), (-60, 52, 0.16, 9), (130, -30, 0.22, 16),
        (40, -70, 0.30, 10), (-150, 30, 0.36, 12), (170, 14, 0.42, 9), (-20, 70, 0.50, 10)]


def iris_masks(cx, cy, R, rot):
    """Mascara em forma de estrela + contornos (externo marrom-dourado, interno amarelo)."""
    P = [(x * SS, y * SS) for x, y in star_points(cx, cy, R, rot, 0.5)]
    out = []
    for kind in ("fill", "outer", "inner"):
        im = Image.new("L", (W * SS, H * SS), 0)
        d = ImageDraw.Draw(im)
        if kind == "fill":
            d.polygon(P, fill=255)
        else:
            lw = (26 if kind == "outer" else 12) * SS
            d.line(P + [P[0], P[1]], fill=255, width=lw, joint="curve")
            r = lw / 2
            for x, y in P[::2]:
                d.ellipse((x - r, y - r, x + r, y + r), fill=255)
        im = im.resize((W, H), Image.BOX)
        out.append(np.asarray(im).astype(np.float32) / 255.0)
    return out


def intro_frame(A, s):
    E = np.zeros((H, W, 3), np.float32)                     # brilho suave
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))  # desenho cartunesco
    d = ImageDraw.Draw(ov)

    x, y, rot, scale, sx, sy = star_state(s)
    R = STAR_R * scale
    final_rot = star_state(REVEAL_START)[2] + 0.5

    if s < REVEAL_START + 0.2:
        pulse = 0.0
        if s >= SHINE_START:
            c = min((s - SHINE_START) / (REVEAL_START - SHINE_START), 1.0)
            pulse = math.sin(math.pi * clamp01(c / 0.6)) if c < 0.6 else 0.0
            pop = 1 + 0.38 * math.sin(math.pi * clamp01(c / 0.45)) if c < 0.45 else 1.0
            if c > 0.6:   # "carrega" (encolhe um pouco) antes de crescer
                pop *= 1 - 0.12 * math.sin(math.pi * clamp01((c - 0.6) / 0.4))
            R *= pop
            rot += 0.5 * ease_in_out(c)
            ra = pulse * 0.9
            if ra > 0.01:   # raios cartunescos girando atras da estrela
                n = 12
                for i in range(n):
                    a = i * 2 * math.pi / n + c * 0.9
                    L0 = R * 1.25
                    L1 = R * (1.9 + (1.3 if i % 2 == 0 else 0.6) * pulse)
                    wdt = 0.09 if i % 2 == 0 else 0.06
                    p1 = (x + L0 * math.cos(a - wdt), y + L0 * math.sin(a - wdt))
                    p2 = (x + L1 * math.cos(a), y + L1 * math.sin(a))
                    p3 = (x + L0 * math.cos(a + wdt), y + L0 * math.sin(a + wdt))
                    d.polygon([(p[0] * SS, p[1] * SS) for p in (p1, p2, p3)],
                              fill=C_LIGHT + (int(220 * ra),))
            for (ox, oy, t0, rr) in POPS:
                q = (c - t0) / 0.35
                if 0 < q < 1:
                    draw_sparkle(d, x + ox, y + oy, rr * math.sin(math.pi * q), q * 1.2, (255, 250, 210))
        add_gauss(E, x, y, R * 1.3, np.array([1.0, 0.85, 0.35], np.float32),
                  0.45 * (0.35 + 0.9 * pulse) * min(scale, 1.0))
        draw_cartoon_star(d, x, y, R, rot, sx, sy)

    for p in PARTS:
        age = s - p["t0"]
        if age < 0 or age > p["life"]:
            continue
        u = age / p["life"]
        if p["drag"]:
            dist = (1 - math.exp(-3.2 * age)) / 3.2
            px, py = p["x"] + p["vx"] * dist, p["y"] + p["vy"] * dist + 60 * age * age
        else:
            px, py = p["x"] + p["vx"] * age, p["y"] + p["vy"] * age
        r = p["r"] * (1 - u) ** 0.8 * min(1.0, age / 0.08)
        if p["kind"] == "sparkle":
            draw_sparkle(d, px, py, r, p["rot"] + p["spin"] * age, C_SPARK)
        else:
            rr = r * 0.32
            d.ellipse(((px - rr) * SS, (py - rr) * SS, (px + rr) * SS, (py + rr) * SS), fill=(255, 250, 220, 230))

    # ---- clarao: a luz da estrela toma a tela toda e depois revela a arte
    flash = None
    if s < FLASH_FULL:
        base = CHROMA * (1 - np.clip(E, 0, 1).max(axis=2, keepdims=True)) + np.clip(E, 0, 1)
        if s >= REVEAL_START:
            c = (s - REVEAL_START) / (FLASH_FULL - REVEAL_START)
            r = 30 + 3000 * c ** 2.2
            dist = np.sqrt((A.xx - STAR_END[0]) ** 2 + (A.yy - STAR_END[1]) ** 2)
            flash = np.clip(1.6 * np.exp(-(dist / r) ** 2), 0, 1)
            if c > 0.9:
                flash = flash + (1 - flash) * smooth(0.9, 1.0, c)
    else:
        base = art_frame(A, s - REVEAL_START, s)
        if s < REVEAL_END:
            q = clamp01((s - FLASH_FADE) / (REVEAL_END - FLASH_FADE))
            flash = np.full((H, W), (1 - q) ** 2.2, np.float32)
            # o centro (onde estava a estrela) clareia por ultimo
            dist = np.sqrt((A.xx - STAR_END[0]) ** 2 + (A.yy - STAR_END[1]) ** 2)
            flash = np.clip(flash * (1 + 0.6 * np.exp(-dist / 500) * (1 - q)), 0, 1)

    rgb = base
    o = np.asarray(ov.convert("RGBa").resize((W, H), Image.BOX)).astype(np.float32) / 255.0
    oa = o[..., 3:4]
    rgb = rgb * (1 - oa) + o[..., :3]          # RGBa = cor pre-multiplicada

    if flash is not None:
        fa = flash[..., None]
        col = WARM_WHITE * (1 - fa) + np.array([1.0, 1.0, 0.97], np.float32) * fa
        rgb = rgb * (1 - fa) + col * fa
    return np.clip(rgb, 0, 1)


def full_frame(A, s):
    """Quadro do video completo no tempo s."""
    if s < INTRO_LEN:
        return intro_frame(A, s)
    return art_frame(A, s - REVEAL_START, s)


def to_u8(a, seed=0):
    n = np.random.default_rng(seed).random(a.shape[:2], dtype=np.float32)[..., None] - 0.5
    return np.clip(a * 255 + n + 0.5, 0, 255).astype(np.uint8)
