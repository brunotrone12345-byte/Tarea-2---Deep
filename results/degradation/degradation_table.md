# Resultados de Robustez ante Degradación en Test (451 imágenes, umbral val = 0.30)

- **Modelo evaluado:** `yolo26s_tl_full` (`runs/yolo/yolo26s_tl_full/weights/best.pt`)

| Condición | Nivel | Parámetro | mAP@0.5 | Δ mAP@0.5 | mAP@0.5:0.95 | Δ mAP@0.5:0.95 | Precisión | Recall | F1 | mAP@0.5 (n>=20) | mAP@0.5 (<32px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sin degradación | Limpio | Original | 0.432 | — | 0.306 | — | 0.503 | 0.394 | 0.442 | 0.609 | 0.396 |
| Baja iluminación | Leve | gamma=1.5 | 0.398 | -8.0% | 0.283 | -7.6% | 0.467 | 0.363 | 0.408 | 0.578 | 0.379 |
| Baja iluminación | Moderado | gamma=2.2 | 0.345 | -20.2% | 0.238 | -22.2% | 0.448 | 0.301 | 0.360 | 0.512 | 0.312 |
| Baja iluminación | Severo | gamma=3.0 | 0.272 | -37.0% | 0.190 | -38.0% | 0.409 | 0.225 | 0.290 | 0.414 | 0.234 |
| Desenfoque gaussiano | Leve | sigma=1.5 px | 0.408 | -5.5% | 0.284 | -7.3% | 0.491 | 0.345 | 0.405 | 0.566 | 0.369 |
| Desenfoque gaussiano | Moderado | sigma=3.0 px | 0.355 | -17.9% | 0.234 | -23.4% | 0.461 | 0.273 | 0.343 | 0.504 | 0.331 |
| Desenfoque gaussiano | Severo | sigma=5.0 px | 0.262 | -39.3% | 0.170 | -44.5% | 0.414 | 0.218 | 0.286 | 0.405 | 0.249 |
| Ruido gaussiano | Leve | sigma=15 | 0.189 | -56.3% | 0.127 | -58.3% | 0.302 | 0.160 | 0.209 | 0.301 | 0.211 |
| Ruido gaussiano | Moderado | sigma=30 | 0.035 | -91.9% | 0.022 | -92.8% | 0.174 | 0.033 | 0.055 | 0.058 | 0.053 |
| Ruido gaussiano | Severo | sigma=50 | 0.010 | -97.6% | 0.005 | -98.4% | 0.074 | 0.008 | 0.014 | 0.016 | 0.001 |
| Bajo contraste / turbidez | Leve | alpha=0.35 (65% contr.) | 0.406 | -6.1% | 0.294 | -3.9% | 0.491 | 0.393 | 0.437 | 0.573 | 0.358 |
| Bajo contraste / turbidez | Moderado | alpha=0.55 (45% contr.) | 0.362 | -16.3% | 0.254 | -16.9% | 0.487 | 0.335 | 0.397 | 0.512 | 0.306 |
| Bajo contraste / turbidez | Severo | alpha=0.75 (25% contr.) | 0.259 | -40.0% | 0.176 | -42.4% | 0.402 | 0.227 | 0.290 | 0.329 | 0.131 |
