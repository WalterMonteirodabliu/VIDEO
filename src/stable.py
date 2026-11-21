"""
Elementos do estabulo, desenhados proceduralmente:
  build_mid()  -> toras do teto, caibros, postes e palha que ficam na arte final
  build_door() -> fachada da entrada (parede de tabuas, batentes, viga, palha)
                  usada na animacao de "entrar no estabulo"
"""
import math

import numpy as np
from PIL import Image, ImageDraw

W, H = 1920, 512
SS = 2


# ---------------------------------------------------------------------------
# ruido
# ---------------------------------------------------------------------------
def _hash(xi, yi, seed):
    h = (xi * 374761393 + yi * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def vnoise(x, y, seed=0):
    xf0, yf0 = np.floor(x), np.floor(y)
    xf, yf = (x - xf0).astype(np.float32), (y - yf0).astype(np.float32)
    xi, yi = xf0.astype(np.int64), yf0.astype(np.int64)
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    a = _hash(xi, yi, seed)
    b = _hash(xi + 1, yi, seed)
    c = _hash(xi, yi + 1, seed)
    d = _hash(xi + 1, yi + 1, seed)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, seed=0, octaves=4):
    s, amp, norm = 0.0, 1.0, 0.0
    for i in range(octaves):
        f = 2 ** i
        s = s + amp * vnoise(x * f, y * f, seed + 17 * i)
        norm += amp
        amp *= 0.5
    return s / norm


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# madeira
# ---------------------------------------------------------------------------
WOOD = np.array([0.46, 0.29, 0.15], np.float32)
CUT = np.array([0.66, 0.48, 0.28], np.float32)


def draw_log(rgb, alpha, p0, p1, r, light, seed=0, cut0=False, cut1=False,
             flat=False, light_side=1.0, tone=1.0):
    """Tora (ou viga, flat=True) de p0 a p1 com raio r.
    v>0 = lado 'direito' da direcao p0->p1 (para toras horizontais: embaixo)."""
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -dy, dx
    m = r + 3
    ch, cw = alpha.shape
    X0 = int(max(min(x0, x1) - m - r * 0.4, 0))
    X1 = int(min(max(x0, x1) + m + r * 0.4, cw))
    Y0 = int(max(min(y0, y1) - m - r * 0.4, 0))
    Y1 = int(min(max(y0, y1) + m + r * 0.4, ch))
    if X1 <= X0 or Y1 <= Y0:
        return
    Y, X = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
    u = (X - x0) * dx + (Y - y0) * dy
    v = ((X - x0) * nx + (Y - y0) * ny) / r

    a_body = np.clip((1 - np.abs(v)) * r, 0, 1)
    if cut0:
        a_body *= np.clip(u + 0.5, 0, 1)
    if cut1:
        a_body *= np.clip(L - u + 0.5, 0, 1)

    if flat:
        cyl = 0.55 + 0.45 * np.clip((1 - np.abs(v)) * 5, 0, 1)
    else:
        cyl = np.sqrt(np.clip(1 - v * v, 0, 1)) ** 0.8
    g1 = fbm(u / 150.0, v * 2.0 + seed, seed)
    lines = 0.5 + 0.5 * np.sin((v * (5 if flat else 8) + g1 * 6) * math.pi)
    bark = fbm(u / 22.0, v * 5.0 + seed * 3, seed + 5)
    cracks = smooth(0.60, 0.70, bark)
    wood = 0.62 + 0.35 * (g1 - 0.5) * 2 * 0.5 + 0.16 * lines - 0.38 * cracks
    side = 0.5 + 0.5 * np.clip(v * light_side, -1, 1)
    shade = (0.16 + 0.84 * cyl) * (0.40 + 0.60 * side)
    Lm = light(X, Y)
    col = WOOD * (shade * wood * 1.5 * tone)[..., None] * Lm[..., None]

    a = a_body
    for flag, uc in ((cut0, 0.0), (cut1, L)):
        if not flag:
            continue
        du = (u - uc) / (0.30 * r)
        rho = np.sqrt(du * du + v * v)
        face = np.clip((1 - rho) * r * 0.45, 0, 1)
        ring = 0.5 + 0.5 * np.sin(rho * 22 + fbm(du * 2, v * 2, seed + 9) * 4)
        fcol = CUT * (0.62 + 0.38 * ring)[..., None] * (0.55 + 0.45 * (1 - rho))[..., None]
        fcol = fcol * (1 - 0.6 * smooth(0.8, 1.0, rho))[..., None]   # casca na borda
        fcol = fcol * Lm[..., None] * tone
        col = col * (1 - face[..., None]) + fcol * face[..., None]
        a = np.maximum(a, face)

    rgb[Y0:Y1, X0:X1] = rgb[Y0:Y1, X0:X1] * (1 - a[..., None]) + col * a[..., None]
    alpha[Y0:Y1, X0:X1] = alpha[Y0:Y1, X0:X1] + a * (1 - alpha[Y0:Y1, X0:X1])


# ---------------------------------------------------------------------------
# palha
# ---------------------------------------------------------------------------
STRAW = np.array([(224, 184, 104), (204, 154, 74), (176, 124, 56), (238, 208, 136),
                  (150, 102, 44), (214, 170, 92)], np.float32) / 255.0


def straw(canvas, rng, n, sample, light, ang_mu=0.0, ang_sd=0.45, length=(10, 40),
          tone=1.0, width=(1.0, 1.8), up_bias=0.0):
    """Desenha n fios de palha num canvas RGBA (2x). sample(rng)->(x,y,profundidade 0..1)."""
    d = ImageDraw.Draw(canvas)
    items = []
    for _ in range(n):
        x, y, depth = sample(rng)
        items.append((depth, x, y))
    items.sort()
    for depth, x, y in items:
        ang = rng.normal(ang_mu, ang_sd) + (math.pi if rng.random() < 0.5 else 0)
        ln = rng.uniform(*length)
        dx, dy = math.cos(ang) * ln, math.sin(ang) * ln * (1 - up_bias) - up_bias * ln * 0.4
        c = STRAW[rng.integers(len(STRAW))]
        b = (0.45 + 0.75 * depth) * float(light(np.float32(x), np.float32(y))) * tone
        b *= rng.uniform(0.8, 1.15)
        col = tuple(int(min(255, v * b * 255)) for v in c) + (255,)
        w = rng.uniform(*width) * SS
        d.line([(x * SS, y * SS), ((x + dx) * SS, (y + dy) * SS)], fill=col, width=max(1, int(round(w))))


def canvas_to_arrays(canvas):
    a = np.asarray(canvas.convert("RGBa").resize((canvas.size[0] // SS, canvas.size[1] // SS),
                                                  Image.BOX)).astype(np.float32) / 255.0
    al = a[..., 3]
    rgb = a[..., :3] / np.maximum(al[..., None], 1e-4)
    return np.clip(rgb, 0, 1), al


def over(rgb, alpha, rgb2, a2):
    rgb[:] = rgb * (1 - a2[..., None]) + rgb2 * a2[..., None]
    alpha[:] = alpha + a2 * (1 - alpha)


# ---------------------------------------------------------------------------
# elementos que ficam na arte (dentro do estabulo)
# ---------------------------------------------------------------------------
def scene_light(baby_x, beam_x):
    def light(X, Y):
        l = 0.30
        l = l + 0.95 * np.exp(-((X - beam_x) / 300.0) ** 2)
        l = l + 0.85 * np.exp(-((X - baby_x) / 320.0) ** 2 - ((Y - 330) / 600.0) ** 2)
        l = l + 0.25 * np.exp(-((X - 1650) / 400.0) ** 2)
        return np.minimum(l, 1.25).astype(np.float32)
    return light


def build_mid(baby_x, beam_x=1080.0, gap=(905, 1225), pad=0):
    """Desenha numa tela maior (margem 'pad' em volta) para a camera poder se mover.
    Coordenadas abaixo sao as da tela 1920x512; o desenho e deslocado por 'pad'."""
    P = pad
    cw, ch = W + 2 * P, H + 2 * P
    base_light = scene_light(baby_x, beam_x)

    def light(X, Y):
        return base_light(X - P, Y - P)

    rgb = np.zeros((ch, cw, 3), np.float32)
    alpha = np.zeros((ch, cw), np.float32)
    rng = np.random.default_rng(21)

    def q(x, y):
        return (x + P, y + P)

    # teto: forro escuro de tabuas + toras acima da tora principal
    if P:
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        roof = np.clip((P - 14 - yy) / 6, 0, 1)
        k = np.floor(yy / 34.0)
        g = fbm(xx / 160.0, yy / 6.0 + k * 3, 41)
        rcol = np.array([0.16, 0.10, 0.05], np.float32) * (0.5 + 0.6 * g)[..., None]
        rcol = rcol * (1 - 0.7 * (np.abs(yy / 34.0 - k - 0.5) > 0.46))[..., None]
        rcol = rcol * light(xx, yy * 0 + 0)[..., None] * 0.8
        over(rgb, alpha, rcol, roof)
        for i, yc in enumerate((-40, -80, -120, -160, -200)):
            draw_log(rgb, alpha, q(-P - 40, yc), q(W + P + 40, yc), 19, light, seed=40 + i,
                     light_side=1, tone=0.75 - 0.08 * i)
        # postes e parede nas laterais (fora da tela)
        for i, xc in enumerate((-150, W + 150)):
            draw_log(rgb, alpha, q(xc, -P - 30), q(xc, H + P + 30), 22, light, seed=50 + i,
                     flat=True, light_side=1 if xc > 0 else -1, tone=0.6)

    # postes nas laterais
    draw_log(rgb, alpha, q(12, -P - 30), q(12, H + P + 30), 20, light, seed=4, flat=True, light_side=-1, tone=0.8)
    draw_log(rgb, alpha, q(W - 12, -P - 30), q(W - 12, H + P + 30), 20, light, seed=6, flat=True, light_side=1, tone=0.8)
    # caibros diagonais nos cantos
    draw_log(rgb, alpha, q(-40 - P, 118 + P * 0.63), q(205, -36), 13, light, seed=8, light_side=1, tone=0.9)
    draw_log(rgb, alpha, q(1690, -36), q(1960 + P, 176 + P * 0.78), 13, light, seed=9, light_side=-1, tone=0.9)
    # tora do teto com um vao por onde entra a luz
    draw_log(rgb, alpha, q(-40 - P, 3), q(gap[0], 3), 19, light, seed=1, cut1=True, light_side=1)
    draw_log(rgb, alpha, q(gap[1], 1), q(W + 40 + P, 1), 19, light, seed=2, cut0=True, light_side=1)

    cv = Image.new("RGBA", (cw * SS, ch * SS), (0, 0, 0, 0))

    def hang(rg):
        while True:
            x = rg.uniform(30 - P, W - 30 + P)
            if not (gap[0] - 10 < x < gap[1] + 10):
                return x + P, rg.uniform(12, 20) + P, rg.random()
    straw(cv, rng, int(260 * cw / W), hang, light, ang_mu=math.pi / 2, ang_sd=0.25, length=(8, 26), tone=0.9)

    piles = [(1700, 230, 72), (640, 95, 42), (60, 130, 34), (1180, 150, 6)]

    def ytop(x):
        y = 512 - 7.0
        for cx, w, h in piles:
            y = min(y, 512 - 7 - h * math.exp(-((x - cx) / w) ** 2))
        return y

    def floor_sample(rg):
        x = rg.uniform(-20 - P, W + 20 + P)
        yt = ytop(x)
        y = rg.uniform(yt, 516 + P)
        depth = 1 - min((y - yt) / 40.0, 1.0)
        return x + P, y + P, depth
    straw(cv, rng, int(5200 * (1 + 1.6 * P / 220) * cw / W), floor_sample, light,
          ang_mu=0.0, ang_sd=0.38, length=(10, 34), tone=1.0)
    r2, a2 = canvas_to_arrays(cv)
    over(rgb, alpha, r2, a2)
    return rgb, alpha


# ---------------------------------------------------------------------------
# fachada da entrada (primeiro plano)
# ---------------------------------------------------------------------------
OPEN = (185, 58, 1735, 452)   # x0, y0, x1, y1 da porta (a arte aparece por dentro)


def build_door():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ox0, oy0, ox1, oy1 = OPEN
    ddx = np.maximum(np.maximum(ox0 - xx, xx - ox1), 0)
    ddy = np.maximum(np.maximum(oy0 - yy, yy - oy1), 0)
    dist = np.sqrt(ddx ** 2 + ddy ** 2)
    outside = np.clip(np.maximum(ddx, ddy), 0, 1)

    def light(X, Y):
        dx = np.maximum(np.maximum(ox0 - X, X - ox1), 0)
        dy = np.maximum(np.maximum(oy0 - Y, Y - oy1), 0)
        d = np.sqrt(dx ** 2 + dy ** 2)
        return (0.22 + 0.95 * np.exp(-d / 75.0)).astype(np.float32)

    # parede de tabuas verticais
    pw = 78.0
    k = np.floor((xx + 23) / pw)
    fx = (xx + 23) / pw - k
    gapline = 1 - (smooth(0.0, 0.035, fx) * smooth(0.0, 0.035, 1 - fx))
    grain = fbm(xx / 7.0 + k * 3.1, yy / 160.0 + k * 7.7, 31)
    streak = 0.5 + 0.5 * np.sin((xx / 5.0 + grain * 9) * 0.9)
    plank_tone = 0.75 + 0.5 * vnoise(k * 1.7, k * 0.3 + 2.0, 77)
    knots = smooth(0.78, 0.84, fbm(xx / 18.0, yy / 18.0, 55, 3))
    wood = (0.55 + 0.45 * grain + 0.12 * streak - 0.35 * knots) * plank_tone
    wall = np.array([0.30, 0.18, 0.09], np.float32) * (wood * (1 - 0.85 * gapline))[..., None]
    wall *= light(xx, yy)[..., None]
    # cantos mais escuros
    vig = 1 - 0.55 * np.clip(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.2)) ** 2, 0, 1)
    wall *= vig[..., None]
    rgb = wall.astype(np.float32)
    alpha = outside.astype(np.float32)

    rng = np.random.default_rng(5)
    # soleira
    draw_log(rgb, alpha, (-40, 472), (W + 40, 472), 18, light, seed=12, light_side=-1, tone=0.75)
    # batentes
    draw_log(rgb, alpha, (ox0 - 22, -40), (ox0 - 22, H + 40), 30, light, seed=13, flat=True, light_side=-1)
    draw_log(rgb, alpha, (ox1 + 22, -40), (ox1 + 22, H + 40), 30, light, seed=14, flat=True, light_side=1)
    # viga de cima (tora)
    draw_log(rgb, alpha, (-40, oy0 - 16), (W + 40, oy0 - 16), 27, light, seed=15, light_side=1)

    cv = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))

    # palha do telhado pendurada sobre a viga
    def thatch(rg):
        x = rg.uniform(-30, W + 30)
        return x, rg.uniform(-30, 14), rg.random()
    straw(cv, rng, 3600, thatch, light, ang_mu=math.pi / 2, ang_sd=0.22, length=(30, 85), tone=0.55,
          width=(1.2, 2.4))

    # palha no chao da entrada
    def ground(rg):
        x = rg.uniform(-30, W + 30)
        yt = 448 + 30 * math.sin(x / 140.0) * 0.5 + rg.normal(0, 6)
        y = rg.uniform(yt, 520)
        return x, y, 1 - (y - yt) / max(520 - yt, 1)
    straw(cv, rng, 4200, ground, light, ang_mu=0.0, ang_sd=0.45, length=(18, 60), tone=0.42,
          width=(1.4, 2.8))
    r2, a2 = canvas_to_arrays(cv)
    over(rgb, alpha, r2, a2)

    a8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    c8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    img = Image.fromarray(np.dstack([c8, a8]), "RGBA")
    return img
