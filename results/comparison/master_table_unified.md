# Tabla Maestra Unificada (Test + Benchmark Uniforme de Inferencia en GPU RTX 5060)

| Modelo | mAP@0.5 | mAP@0.5:0.95 | mAP@0.75 | Precisión | Recall | F1 | mAP@0.5 (n>=20) | Params (M) | Entrenamiento (min) | Inferencia GPU (ms) | p95 GPU (ms) | FPS GPU | VRAM pico (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN propia | 0.258 | 0.124 | 0.099 | 0.267 | 0.201 | 0.230 | 0.416 | 13.13 | 55.5 | 10.25 | 11.92 | 97.6 | 119.9 |
| YOLO26s desde cero | 0.284 | 0.165 | 0.173 | 0.364 | 0.254 | 0.299 | 0.452 | 9.97 | 65.9 | 13.63 | 17.87 | 73.4 | 123.5 |
| YOLO26s transfer | 0.404 | 0.307 | 0.335 | 0.425 | 0.374 | 0.398 | 0.610 | 9.97 | 22.6 | 12.71 | 16.69 | 78.7 | 161.9 |
| Faster R-CNN transfer | 0.373 | 0.228 | 0.265 | 0.345 | 0.372 | 0.358 | 0.516 | 41.44 | 33.8 | 36.60 | 35.73 | 27.3 | 463.2 |

*Nota:* La latencia GPU corresponde al protocolo uniforme (`batch=1`, `fp32`, imagen BGR en RAM -> cajas finales con NMS en CPU, 150 imágenes tras calentamiento de 20).
