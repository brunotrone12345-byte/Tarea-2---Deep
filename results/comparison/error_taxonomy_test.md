# Desglose Cuantitativo de Errores en Test (sobre 1.004 objetos reales)

| modelo | umbral_val | TP_correcto | FP_confusion_especie | FP_mala_localizacion | FP_loc_y_especie | FP_duplicado | FP_fondo | FN_no_detectado |
|---|---|---|---|---|---|---|---|---|
| CNN propia | 0.3 | 459 | 142 | 29 | 9 | 1 | 120 | 545 |
| YOLO26s desde cero | 0.3 | 463 | 112 | 34 | 14 | 10 | 74 | 541 |
| YOLO26s transfer | 0.3 | 623 | 113 | 24 | 10 | 10 | 85 | 381 |
| Faster R-CNN transfer | 0.35 | 585 | 207 | 34 | 19 | 5 | 139 | 419 |
