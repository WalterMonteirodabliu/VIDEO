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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import stable

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
    "sub":   (4.50, 0.80),   # "Especial de Natal" (maquina de escrever)
    "title": (4.95, 3.00),   # "O Autor da Historia" escrito por uma pena dourada
    "date":  (7.55, 0.90),   # "20 de dezembro | 19h" (maquina de escrever)
    "logo":  (5.80, 0.80),
    "line":  (6.15, 0.65),
    "t1":    (6.30, 0.65),
    "t2":    (6.45, 0.65),
    "t3":    (6.60, 0.65),
}
ENTR_DONE = 8.6

# ---- camera entrando no estabulo (atravessa a porta)
DOLLY_START = 3.15
DOLLY_END = 5.0
DOOR_END_SCALE = 2.6       # quanto a porta (primeiro plano) cresce
SCENE_DEPTH = 3.0          # distancia relativa da cena em relacao a porta (paralaxe)
PAD = 220                  # margem (espelhada) da arte para o movimento de camera

CHROMA = np.array([0.0, 1.0, 0.0], np.float32)  # verde puro #00FF00
GOLD = np.array([1.0, 0.80, 0.52], np.float32)
WARM_WHITE = np.array([1.0, 0.95, 0.85], np.float32)

STAR_START = (960.0, 470.0)
STAR_END = (952.0, 78.0)     # onde a luz do ceu entra na arte
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

    # ---- fundo escuro quente (calculado tambem na margem usada pela camera)
    def background(X, Y):
        bg = np.zeros(X.shape + (3,), np.float32)
        bg[:] = np.array([0.020, 0.013, 0.007], np.float32)
        h1 = np.exp(-(((X - 960) / 620) ** 2 + ((Y - 230) / 380) ** 2))
        bg += h1[..., None] * np.array([0.060, 0.036, 0.015], np.float32)
        h2 = np.exp(-(((X - 340) / 520) ** 2 + ((Y - 250) / 320) ** 2))
        bg += h2[..., None] * np.array([0.022, 0.014, 0.006], np.float32)
        return bg
    base = background(xx, yy)
    ey, ex = np.mgrid[-PAD:H + PAD, -PAD:W + PAD].astype(np.float32)
    A.ext = background(ex, ey)

    # ---- presepio no centro, embaixo da luz
    nat = flyer[780:1400, 150:880]
    nh = H + 8
    sc = nh / 620.0
    nw = int(round(nat.shape[1] * sc))
    nat = resize(nat, nw, nh)
    nm = feather_mask(nh, nw, left=130, right=130, top=60, bottom=0)
    A.nat_x0 = 960 - nw // 2
    paste(base, nat, nm, A.nat_x0, -4)
    A.baby = (A.nat_x0 + (430 - 150) * sc, -4 + (1145 - 780) * sc)

    # ---- titulo dourado recortado do flyer (sem o feixe de luz de tras)
    ttl = flyer[240:770, 10:890]
    tw = int(round(ttl.shape[1] * H / ttl.shape[0]))
    ttl = resize(ttl, tw, H)
    m, plate = title_matte(ttl)
    r0, r1 = 66, 436
    st = 0.80
    pw, ph_ = int(round(tw * st)), int(round((r1 - r0) * st))
    prgb = resize(ttl[r0:r1], pw, ph_)
    pa = np.asarray(Image.fromarray((m[r0:r1] * 255).astype(np.uint8)).resize((pw, ph_), Image.LANCZOS)
                    ).astype(np.float32) / 255.0
    TX, TY = 350, 252
    x0, y0 = TX - pw // 2, TY - ph_ // 2
    # feixe de luz original do flyer atras do titulo (coluna inteira, sem os textos)
    beam_bg = resize(np.clip(plate, 0, 1), pw, H)
    paste(base, beam_bg, feather_mask(H, pw, left=50, right=50, top=0, bottom=0), x0, 0)
    paste(base, nat, nm, A.nat_x0, -4)
    # titulo original (letras + feixe) por cima
    A.title_rgb = np.zeros((H, W, 3), np.float32)
    A.title_a = np.zeros((H, W), np.float32)      # recorte do titulo com bordas suaves
    A.title_m = np.zeros((H, W), np.float32)      # so as letras (para a pena revelar)
    A.title_rgb[y0:y0 + ph_, x0:x0 + pw] = prgb
    A.title_a[y0:y0 + ph_, x0:x0 + pw] = feather_mask(ph_, pw, left=50, right=50, top=24, bottom=24)
    A.title_m[y0:y0 + ph_, x0:x0 + pw] = pa
    Lt = lum(A.title_rgb)
    A.letters = (A.title_m * _smooth(0.40, 0.62, Lt)).astype(np.float32)

    # linhas do titulo ("O Autor" / "da" / "Historia") para a pena escrever
    cuts = [(70, 212), (212, 262), (262, 432)]
    A.lines = []
    for a0, a1 in cuts:
        ya, yb = y0 + (a0 - r0) * st, y0 + (a1 - r0) * st
        rows = A.title_m[int(ya):int(yb)]
        cols = np.where(rows.max(axis=0) > 0.5)[0]
        A.lines.append(dict(y0=ya, y1=yb, x0=float(cols.min()), x1=float(cols.max())))
    A.line_w = []
    yv = np.arange(H, dtype=np.float32)
    for i, ln in enumerate(A.lines):
        lo = -1e9 if i == 0 else ln["y0"]
        hi = 1e9 if i == len(A.lines) - 1 else ln["y1"]
        A.line_w.append((_smooth(lo - 3, lo + 3, yv) * (1 - _smooth(hi - 3, hi + 3, yv)))[:, None])

    # subtitulo e data (maquina de escrever)
    f_sub = ImageFont.truetype(os.path.join(FONT_DIR, "Inter-SemiBold.otf"), 31)
    f_date = ImageFont.truetype(os.path.join(FONT_DIR, "Inter-Bold.otf"), 38)
    A.sub = TypeSprite("Especial de Natal", f_sub, (226, 194, 152, 255), TX, 80)
    A.date = TypeSprite("20 de dezembro | 19h", f_date, (255, 255, 255, 255), TX, 440)

    # ---- coluna da direita
    cx = 1598
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
    A.door = stable.build_door().convert("RGBa")

    vx = (xx - W / 2) / (W / 2)
    vy = (yy - H / 2) / (H / 2)
    A.vig = (1 - 0.35 * np.clip(vx ** 2 * 0.6 + vy ** 2 * 0.25, 0, 1))[..., None].astype(np.float32)

    # ---- feixe de luz do ceu (geometria em coordenadas da arte)
    A.beam_o = (1040.0, -420.0)
    bx, by = A.baby
    A.beam_th0 = math.atan2(bx - A.beam_o[0], by - A.beam_o[1])
    A.beam_len = math.hypot(bx - A.beam_o[0], by - A.beam_o[1])
    A.beam_top = (A.beam_o[0] + math.tan(A.beam_th0) * (0 - A.beam_o[1]), 0.0)

    # ---- arte final (tudo ja entrou)
    A_EXT[0] = A.ext
    A.art = compose(A, None)
    A.art_img = pad_img(A.art)

    cand = np.argwhere((A.letters > 0.9) & (lum(A.art) > 0.80))
    rng = np.random.default_rng(7)
    rng.shuffle(cand)
    picks = []
    for y, x in cand:
        if all((x - px) ** 2 + (y - py) ** 2 > 120 ** 2 for py, px in picks):
            picks.append((y, x))
        if len(picks) >= 6:
            break
    A.glints = [(float(x), float(y)) for y, x in picks]

    # ---- poeira dourada (principalmente dentro do feixe)
    rng = np.random.default_rng(3)
    n = 240
    grp = rng.random(n)
    along = rng.uniform(0.0, 1.05, n)
    ox, oy = A.beam_top
    px_beam = ox + (bx - ox) * along + rng.normal(0, 38, n) * (0.6 + along)
    py_beam = oy + (by - oy) * along
    A.dust = dict(
        x=np.where(grp < 0.7, px_beam, rng.uniform(0, W, n)).astype(np.float32),
        y=np.where(grp < 0.7, py_beam, rng.uniform(-20, H + 20, n)).astype(np.float32),
        phase=rng.random(n).astype(np.float32),
        life_div=rng.choice([1, 2], n),
        vy=rng.uniform(-14, -4, n).astype(np.float32),
        sway=rng.uniform(4, 16, n).astype(np.float32),
        sway_k=rng.choice([1, 2, 3], n),
        size=rng.uniform(0.7, 2.2, n).astype(np.float32),
        bri=(rng.uniform(0.3, 1.0, n) * np.where(grp < 0.7, 1.0, 0.35)).astype(np.float32),
    )
    return A


def _blur_ax(a, r, ax, passes=3):
    for _ in range(passes):
        p = np.concatenate([np.repeat(a.take([0], ax), r + 1, ax), a,
                            np.repeat(a.take([-1], ax), r, ax)], ax)
        c = np.cumsum(p, axis=ax, dtype=np.float64)
        n = a.shape[ax]
        a = ((c.take(range(2 * r + 1, 2 * r + 1 + n), ax) - c.take(range(0, n), ax)) / (2 * r + 1)).astype(np.float32)
    return a


def title_matte(ttl):
    """Recorta as letras douradas do titulo, separando-as do feixe de luz do flyer."""
    th = ttl.shape[0]
    ty = np.arange(th, dtype=np.float32)[:, None]
    L = lum(ttl)
    top = box_blur(ttl[4:14].mean(0, keepdims=True), 4, 2)
    bot = box_blur(ttl[503:511].mean(0, keepdims=True), 4, 2)
    v = np.clip((ty - 9) / 498, 0, 1)[..., None]
    plate = top * (1 - v) + bot * v
    band = ((ty > 70) & (ty < 425)).astype(np.float32) * np.ones_like(L)
    for _ in range(4):   # estima o feixe ignorando as letras
        Lp = lum(plate)
        letter = ((L - Lp) > 0.07) | ((Lp - L) > 0.05)
        letter = box_blur(letter.astype(np.float32), 2, 1) > 0.05
        w = (~letter).astype(np.float32)
        num = _blur_ax(_blur_ax(ttl * w[..., None], 60, 0), 4, 1)
        den = _blur_ax(_blur_ax(w, 60, 0), 4, 1)
        plate = num / np.maximum(den, 1e-4)[..., None]
    Lp = lum(plate)
    bw = _smooth(0.08, 0.30, Lp)
    core = ((L - Lp) > 0.10 + 0.22 * bw) & (band > 0)
    near = box_blur(core.astype(np.float32), 3, 1) > 0.02
    dark = ((Lp - L) > 0.03 + 0.05 * bw) & near
    m = (core | dark | (near & ((L - Lp) > 0.045 + 0.2 * bw))).astype(np.float32)
    return np.clip(box_blur(m, 1, 1) * 1.3, 0, 1) * band, plate


class TypeSprite:
    """Texto que aparece letra por letra (maquina de escrever)."""

    def __init__(self, txt, font, color, cx, cy):
        self.spr = text_sprite(txt, font, color, cx, cy)
        w, h = self.spr.img.size
        total = font.getlength(txt)
        start = (w - total) / 2
        self.bounds = [start + font.getlength(txt[:i]) for i in range(len(txt) + 1)]
        self.n = len(txt)
        a = np.asarray(self.spr.img).astype(np.float32) / 255.0
        self.rgb, self.a = a[..., :3], a[..., 3]
        self.x0 = int(round(cx - w / 2))
        self.y0 = int(round(cy - h / 2))
        self.h = h

    def draw(self, f, p, t_abs=0.0):
        if p <= 0:
            return
        nv = p * self.n
        cols = np.zeros(self.rgb.shape[1], np.float32)
        for i in range(self.n):
            vis = clamp01((nv - i) * 2.5)
            if vis <= 0:
                break
            b0, b1 = int(round(self.bounds[i])), int(round(self.bounds[i + 1]))
            cols[b0:b1 + (2 if i == self.n - 1 else 0)] = vis
        cols[:int(round(self.bounds[0]))] = 1.0 if nv > 0 else 0.0
        if p >= 1:
            cols[:] = 1.0
        paste(f, self.rgb, self.a * cols[None, :], self.x0, self.y0)
        # cursor piscando enquanto digita
        if p < 1:
            k = int(min(nv, self.n))
            xc = self.x0 + self.bounds[k] + 3
            on = 1.0 if (t_abs * 2.2) % 1.0 < 0.65 else 0.25
            ys = slice(self.y0 + 4, self.y0 + self.h - 4)
            xi = int(xc)
            f[ys, xi:xi + 3] = f[ys, xi:xi + 3] * (1 - on) + GOLD * on


def entr(s, key):
    if s is None:
        return 1.0
    t0, d = ENTR[key]
    return clamp01((s - t0) / d)


# fases da entrada do titulo (fracao de ENTR["title"]): pena chega, escreve, sai
Q_IN, Q_OUT = 0.08, 0.88
LINE_SPANS = [(0.00, 0.40), (0.46, 0.56), (0.62, 1.00)]   # dentro da fase de escrita


def quill_state(A, p):
    """Ponta da pena (x, y), quanto cada linha ja foi escrita e visibilidade da pena."""
    q = clamp01((p - Q_IN) / (Q_OUT - Q_IN))
    fronts = []
    tip = None
    for i, (a, b) in enumerate(LINE_SPANS):
        ln = A.lines[i]
        u = clamp01((q - a) / (b - a))
        fx = ln["x0"] - 12 + (ln["x1"] - ln["x0"] + 30) * u
        fronts.append(fx if q > a else -1e9)
        if a <= q <= b:
            yc = (ln["y0"] + ln["y1"]) / 2
            amp = (ln["y1"] - ln["y0"]) * 0.32
            tip = (fx, yc + amp * math.sin(u * (ln["x1"] - ln["x0"]) / 26.0))
    if tip is None:   # entre linhas: volta para o inicio da proxima
        for i in range(len(LINE_SPANS) - 1):
            a, b = LINE_SPANS[i][1], LINE_SPANS[i + 1][0]
            if a < q < b:
                u = ease_in_out((q - a) / (b - a))
                l0, l1 = A.lines[i], A.lines[i + 1]
                xa, ya = l0["x1"] + 18, (l0["y0"] + l0["y1"]) / 2
                xb, yb = l1["x0"] - 12, (l1["y0"] + l1["y1"]) / 2
                tip = (xa + (xb - xa) * u, ya + (yb - ya) * u - 40 * math.sin(math.pi * u))
    l0, l2 = A.lines[0], A.lines[-1]
    if p < Q_IN:      # pena chegando
        u = ease_out_cubic(p / Q_IN)
        sx, sy = l0["x0"] - 12, (l0["y0"] + l0["y1"]) / 2
        tip = (sx - 160 * (1 - u), sy - 120 * (1 - u))
        vis = u
    elif p > Q_OUT:   # pena indo embora
        u = ease_in_out((p - Q_OUT) / (1 - Q_OUT))
        sx, sy = l2["x1"] + 18, (l2["y0"] + l2["y1"]) / 2
        tip = (sx + 200 * u, sy - 260 * u)
        vis = 1 - smooth(0.3, 1.0, u)
    else:
        vis = 1.0
    return tip, fronts, vis


def draw_quill(f, tip, ang, vis, scale=1.0):
    """Pena dourada; tip = ponta que escreve, ang = inclinacao (rad)."""
    if vis <= 0.01:
        return
    L = 175 * scale
    S = 3
    size = int(L * 1.25)
    img = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ca, sa = math.cos(ang), math.sin(ang)
    ox, oy = size * 0.12, size * 0.88    # ponta no canto inferior esquerdo do sprite

    def P(u, v):  # u ao longo da pena, v perpendicular
        return ((ox + u * ca - v * sa) * S, (oy + u * sa + v * ca) * S)
    # pena (vane) assimetrica
    top, bot = [], []
    n = 40
    for i in range(n + 1):
        t = i / n
        u = L * (0.22 + 0.78 * t)
        w = math.sin(math.pi * t) ** 0.75
        curve = -6 * scale * math.sin(math.pi * t)
        top.append(P(u, curve - 24 * scale * w * (1.0 - 0.25 * t)))
        bot.append(P(u, curve + 15 * scale * w))
    d.polygon(top + bot[::-1], fill=(212, 160, 72, 255))
    # luz na metade de cima
    mid = [P(L * (0.22 + 0.78 * i / n), -6 * scale * math.sin(math.pi * i / n)) for i in range(n + 1)]
    d.polygon(top + mid[::-1], fill=(246, 214, 140, 255))
    # barbas
    for i in range(3, n - 2, 2):
        t = i / n
        u = L * (0.22 + 0.78 * t)
        w = math.sin(math.pi * t) ** 0.75
        curve = -6 * scale * math.sin(math.pi * t)
        d.line([P(u, curve), P(u + 14 * scale, curve - 22 * scale * w)], fill=(176, 120, 44, 255), width=S)
        d.line([P(u, curve), P(u + 10 * scale, curve + 13 * scale * w)], fill=(170, 116, 42, 255), width=S)
    # haste
    d.line([P(L * 0.08, 0), P(L * 1.02, -4 * scale)], fill=(255, 236, 190, 255), width=int(2.2 * S * scale))
    # bico de metal
    d.polygon([P(0, 0), P(L * 0.20, -6 * scale), P(L * 0.24, 0), P(L * 0.20, 6 * scale)], fill=(120, 84, 34, 255))
    d.polygon([P(0, 0), P(L * 0.20, -6 * scale), P(L * 0.22, -1 * scale)], fill=(236, 200, 120, 255))
    img = img.resize((size, size), Image.LANCZOS)
    a = np.asarray(img.convert("RGBa")).astype(np.float32) / 255.0
    X0, Y0 = int(round(tip[0] - ox)), int(round(tip[1] - oy))
    h, w = a.shape[:2]
    dx0, dy0 = max(X0, 0), max(Y0, 0)
    dx1, dy1 = min(X0 + w, W), min(Y0 + h, H)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sub = a[dy0 - Y0:dy1 - Y0, dx0 - X0:dx1 - X0]
    al = sub[..., 3:4] * vis
    f[dy0:dy1, dx0:dx1] = f[dy0:dy1, dx0:dx1] * (1 - al) + sub[..., :3] * vis


def compose(A, s):
    """Monta a arte com as entradas no tempo absoluto s."""
    f = A.base.copy()

    # "Especial de Natal": maquina de escrever
    A.sub.draw(f, entr(s, "sub"), s or 0.0)

    # titulo: escrito por uma pena dourada, linha por linha
    p = entr(s, "title")
    if True:
        if p >= 1:
            rev = np.ones((H, W), np.float32)
            fronts = None
        elif p <= 0:
            rev = np.zeros((H, W), np.float32)
            fronts = None
        else:
            tip, fronts, vis = quill_state(A, p)
            xs = A.xx[0][None, :]
            rev = np.zeros((H, W), np.float32)
            for wgt, fx in zip(A.line_w, fronts):
                rev += wgt * _smooth(fx + 4, fx - 26, xs)
        # antes da pena passar so aparece o feixe; atras dela, o titulo original
        a = A.title_a * rev
        f = f * (1 - a[..., None]) + A.title_rgb * a[..., None]
        if fronts is not None:
            # tinta dourada fresca brilhando atras da pena
            ink = np.zeros((H, W), np.float32)
            for wgt, fx in zip(A.line_w, fronts):
                ink += wgt * np.exp(-((xs - fx + 18) / 30) ** 2)
            f += (ink * A.letters)[..., None] * np.array([1.0, 0.82, 0.5], np.float32) * 0.9
            if Q_IN <= p <= Q_OUT and tip is not None:
                rng = np.random.default_rng(int(p * 100000))
                add_gauss(f, tip[0], tip[1], 9, GOLD, 0.5)
                for _ in range(4):
                    add_star(f, tip[0] + rng.normal(0, 9), tip[1] + rng.normal(0, 9), core=1.1, glow=3,
                             spike_len=rng.uniform(8, 20), spike_amp=0.8, rot=rng.uniform(0, 1),
                             color=WARM_WHITE, win=26)
            if tip is not None:
                wob = 0.10 * math.sin(p * 60)
                draw_quill(f, tip, -0.95 + wob, vis)

    # data: maquina de escrever
    A.date.draw(f, entr(s, "date"), s or 0.0)

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


def beam_light(A, z, ph, amb):
    """Feixe de luz do ceu sobre o presepio, brilhando e pulsando."""
    cx, cy = 960, 240
    ox, oy = cx + (A.beam_o[0] - cx) * z, cy + (A.beam_o[1] - cy) * z
    X0, X1 = 520, 1400
    xx = A.xx[:, X0:X1]
    yy = A.yy[:, X0:X1]
    th = np.arctan2(xx - ox, yy - oy)
    d = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2) / z
    dth = th - A.beam_th0
    w = 0.085
    cone = np.exp(-(dth / w) ** 2)
    core = np.exp(-(dth / (w * 0.42)) ** 2)
    streak = (0.62 + 0.22 * np.sin(th * 170 + ph) + 0.12 * np.sin(th * 390 - 2 * ph)
              + 0.08 * np.sin(th * 83 + 3 * ph))
    Lb = A.beam_len
    fall = 1 - 0.8 * _smooth(Lb - 40, Lb + 150, d)
    top = 1 + 0.5 * (1 - _smooth(430, Lb, d))
    pulse = 0.86 + 0.14 * math.sin(2 * ph) + 0.05 * math.sin(5 * ph + 1)
    I = (0.26 * cone * streak + 0.24 * core) * fall * top * pulse * amb
    out = np.zeros((H, W), np.float32)
    out[:, X0:X1] = I
    return out, pulse


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
    amb = 1.0 if s is None else 0.25 + 0.75 * smooth(DOLLY_START, DOLLY_END, s)
    fs, zd = dolly(s)
    deco = 1.0 if s is None else smooth(ENTR_DONE, ENTR_DONE + 0.5, s)

    im = pad_img(compose(A, s)) if entering else A.art_img
    z = (1.006 + 0.012 * (0.5 - 0.5 * math.cos(ph))) * zd   # nunca 1.0: nitidez constante
    cx, cy = 960, 240
    im = im.transform((W, H), Image.AFFINE,
                      (1 / z, 0, cx - cx / z + PAD, 0, 1 / z, cy - cy / z + PAD),
                      resample=Image.BICUBIC)
    f = np.asarray(im).astype(np.float32) / 255.0

    glow = (0.12 + 0.06 * math.sin(ph * 2)) * amb
    bx, by = A.baby
    bx, by = cx + (bx - cx) * z, cy + (by - cy) * z
    g = np.exp(-(((A.xx - bx) / 170) ** 2 + ((A.yy - by) / 110) ** 2))
    f += g[..., None] * GOLD * glow

    rays, pulse = beam_light(A, z, ph, amb)
    light = rays[..., None] * np.array([1.0, 0.86, 0.60], np.float32)
    f = 1 - (1 - f) * (1 - np.clip(light, 0, 1))
    # brilho na entrada da luz, no alto
    tx, ty = cx + (A.beam_top[0] - cx) * z, cy + (A.beam_top[1] - cy) * z
    tg = np.exp(-(((A.xx - tx) / 150) ** 2 + ((A.yy - ty) / 70) ** 2))
    f += tg[..., None] * np.array([1.0, 0.85, 0.6], np.float32) * (0.22 * pulse * amb)
    add_star(f, tx, ty + 6, core=2.5, glow=10, spike_len=60 + 40 * pulse, spike_amp=0.45 * pulse * amb,
             rot=0.0, color=WARM_WHITE, win=110)

    if deco > 0:
        sweep_t = (t - 1.0) / 2.4
        if 0 <= sweep_t <= 1:
            pos = -300 + sweep_t * 1600
            band = np.exp(-((A.xx - 20 - pos * 0.45 - (A.yy - 256) * 0.45) / 60) ** 2)
            f += (band * A.letters * deco)[..., None] * np.array([0.55, 0.45, 0.30], np.float32)
        for i, (gx, gy) in enumerate(A.glints):
            c = (t - (3.6 + i * 1.05)) / 0.9
            if 0 < c < 1:
                k = math.sin(math.pi * c) ** 2 * deco
                add_star(f, gx, gy, core=1.6, glow=6, spike_len=55 * k + 5, spike_amp=0.9 * k,
                         rot=0.3 * c, color=WARM_WHITE)

    dust_layer(A, t, f, amb)

    # porta do estabulo em primeiro plano (so na entrada)
    if fs is not None and fs < DOOR_END_SCALE * 0.999:
        dimg = A.door.transform((W, H), Image.AFFINE,
                                (1 / fs, 0, cx - cx / fs, 0, 1 / fs, cy - cy / fs),
                                resample=Image.BICUBIC)
        blur = 0.3 + 3.5 * (fs - 1) / (DOOR_END_SCALE - 1)
        dimg = dimg.filter(ImageFilter.GaussianBlur(blur))
        da = np.asarray(dimg).astype(np.float32) / 255.0      # RGBa (pre-multiplicado)
        f = f * (1 - da[..., 3:4]) + da[..., :3]
    f *= A.vig
    return f


def dolly(s):
    """(escala da porta, escala da cena) no tempo s. Paralaxe de uma camera que avanca."""
    dmax = 1 - 1 / DOOR_END_SCALE
    z0 = (SCENE_DEPTH - dmax) / SCENE_DEPTH
    if s is None or s >= DOLLY_END:
        return None, 1.0
    u = ease_in_out((s - DOLLY_START) / (DOLLY_END - DOLLY_START))
    dz = u * dmax                      # quanto a camera andou
    fs = 1 / (1 - dz)
    zd = z0 * SCENE_DEPTH / (SCENE_DEPTH - dz)
    return fs, zd


def pad_img(a):
    """Arte dentro do estabulo maior (para nao aparecer borda durante o movimento)."""
    p = A_EXT[0].copy()
    p[PAD:PAD + H, PAD:PAD + W] = a
    return to_img(p)


A_EXT = [None]


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
