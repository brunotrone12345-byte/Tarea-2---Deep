# Desempeño por Tamaño de Objeto en Test (Escala de referencia 640 px)

| Modelo | mAP@0.5 Peq (<32px, n=384) | mAP@0.5 Med (32-96px, n=355) | mAP@0.5 Gra (>=96px, n=265) | mAP@0.5:0.95 Peq | mAP@0.5:0.95 Med | mAP@0.5:0.95 Gra | Recall Peq | Recall Med | Recall Gra |
|---|---|---|---|---|---|---|---|---|---|
| CNN propia | 0.275 | 0.282 | 0.321 | 0.133 | 0.141 | 0.152 | 0.393 | 0.510 | 0.479 |
| YOLO26s desde cero | 0.300 | 0.262 | 0.376 | 0.183 | 0.155 | 0.240 | 0.258 | 0.493 | 0.713 |
| YOLO26s transfer | 0.425 | 0.384 | 0.494 | 0.315 | 0.291 | 0.412 | 0.445 | 0.665 | 0.815 |
| Faster R-CNN transfer | 0.286 | 0.318 | 0.507 | 0.157 | 0.197 | 0.344 | 0.354 | 0.628 | 0.853 |
