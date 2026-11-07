| modelo | mAP@0.5 test | mAP@0.5:0.95 test | mAP@0.75 test | precision test | recall test | F1 test | mAP@0.5 test (clases con n>=20) | mAP@0.5:0.95 test (clases con n>=20) | mAP@0.5 val | mAP@0.5:0.95 val | fitness val | params (M) | epocas | mejor epoca | tiempo total (min) | ms/imagen (provisional) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN propia | 0.258 | 0.124 | 0.099 | 0.267 | 0.201 | 0.229 | 0.416 | 0.201 | 0.329 | 0.147 | 0.165 | 13.1 | 100 | 95 | 55.5 | 8.923 |
| YOLO26s desde cero | 0.284 | 0.165 | 0.173 | 0.365 | 0.254 | 0.300 | 0.452 | 0.274 | 0.329 | 0.189 | 0.203 | 9.970 | 150 | 135 | 65.9 | 14.8 |
| YOLO26s transfer | 0.404 | 0.307 | 0.335 | 0.425 | 0.374 | 0.398 | 0.610 | 0.468 | 0.498 | 0.362 | 0.376 | 9.970 | 50 | 37 | 22.6 | 16.1 |
| Faster R-CNN transfer | 0.373 | 0.228 | 0.265 | 0.345 | 0.372 | 0.358 | 0.516 | 0.325 | 0.408 | 0.252 | 0.268 | 41.4 | 20 | 20 | 33.8 | 27.8 |

Ablación (solo val): YOLO26s transfer (backbone congelado): mAP@0.5 = 0.421, mAP@0.5:0.95 = 0.305, fitness = 0.317, tiempo total = 18.3 min.
