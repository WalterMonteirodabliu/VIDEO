# O Autor da História — vídeo para o telão (1920x512)

Arquivos em `output/`:

| Arquivo | O que é |
|---|---|
| `O_Autor_da_Historia_Telao_1920x512.mp4` | Vídeo completo (62,5 s): estrela cartunesca sobe sobre fundo verde (#00FF00) → brilha → clarão de luz toma a tela e se dissipa mostrando a entrada do estábulo → câmera atravessa a porta e entra → textos entram animados → arte em loop. O último quadro emenda direto no início do `Loop_Arte` |
| `Loop_Arte_1920x512.mp4` | Só a arte animada, 10 s, loop perfeito (pode repetir infinito) |
| `Arte_Telao_1920x512.png` | Arte estática do telão |

Linha do tempo do vídeo completo: 0–1,25 s estrela sobe girando · 1,25–1,55 s quica no topo · 1,55–2,45 s brilha (raios + brilhinhos) · 2,45–2,8 s clarão de luz cobre a tela · 2,9–3,8 s clarão se dissipa mostrando a porta do estábulo · 3,15–5,0 s câmera entra no estábulo (porta passa pela câmera com paralaxe) · 4,55–7,7 s entradas: "Especial de Natal" desce, título é escrito por um traço de luz, data sobe, logo cresce, filete abre do centro, endereço sobe em sequência · depois, arte em loop (ciclos de 10 s).

Para regerar: `cd src && python3 render.py` (precisa de Python 3 + numpy + Pillow + ffmpeg).

A arte tem elementos de estábulo desenhados por código (`src/stable.py`): toras no teto com vão para a luz, caibros, postes e palha no chão. Na entrada aparece a fachada (parede de tábuas, batentes, viga e palha do telhado).
