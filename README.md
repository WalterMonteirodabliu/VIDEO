# O Autor da História — vídeo para o telão (1920x512)

Arquivos em `output/`:

| Arquivo | O que é |
|---|---|
| `O_Autor_da_Historia_Telao_1920x512.mp4` | Vídeo completo (64 s): estrela sobe sobre fundo verde (#00FF00) → brilha → expande e revela a arte → arte animada em loop |
| `Loop_Arte_1920x512.mp4` | Só a arte animada, 10 s, loop perfeito (pode repetir infinito) |
| `Arte_Telao_1920x512.png` | Arte estática do telão |

Linha do tempo do vídeo completo: 0–3 s estrela subindo · 3–4 s brilho · 4–5,5 s expansão/revelação · 5,5 s em diante arte em loop (ciclos de 10 s).

Para regerar: `cd src && python3 render.py` (precisa de Python 3 + numpy + Pillow + ffmpeg).
