# O Autor da História — vídeo para o telão (1920x512)

Arquivos em `output/`:

| Arquivo | O que é |
|---|---|
| `O_Autor_da_Historia_Telao_1920x512.mp4` | Vídeo completo (62,5 s). O último quadro emenda direto no início do `Loop_Arte` |
| `Loop_Arte_1920x512.mp4` | Só a arte animada, 10 s, loop perfeito (pode repetir infinito) |
| `Arte_Telao_1920x512.png` | Arte estática do telão |

Layout: título "O Autor da História" à esquerda (com "Especial de Natal" e a data), presépio no centro sob a luz que vem do céu, logo e endereço à direita.

Linha do tempo do vídeo completo:
- 0–1,25 s estrela cartunesca sobe girando sobre fundo verde (#00FF00) · 1,25–1,55 s quica no topo · 1,55–2,45 s brilha
- 2,45–2,8 s clarão de luz cobre a tela · 2,9–3,8 s clarão se dissipa mostrando a porta do estábulo
- 3,15–5,0 s câmera atravessa a porta e entra no estábulo
- 4,5 s "Especial de Natal" datilografado · 4,95–7,9 s uma pena dourada escreve o título linha por linha · 5,8–7,3 s logo, filete e endereço entram · 7,55 s data datilografada
- depois: arte em loop (ciclos de 10 s) — feixe de luz do céu pulsando, poeira dourada, brilho passando nas letras

Para regerar: `cd src && python3 render.py` (precisa de Python 3 + numpy + Pillow + ffmpeg). `src/stable.py` desenha a fachada do estábulo usada na entrada.
