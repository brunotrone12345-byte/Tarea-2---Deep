# Detección de Organismos Marinos en Imágenes Submarinas (*FjordFish*)

**Asignatura:** CIT3360 — Deep Learning (2026-2), Universidad Diego Portales  
**Evaluación:** Tarea 2 — Comparación de arquitecturas desde cero, *Transfer Learning* y robustez ante degradación visual

---

## Estructura del Repositorio

```text
├── data/fjord/                 # Dataset FjordFish en formato YOLO (2.075 train / 476 val / 451 test)
│   └── data.yaml               # Configuración portable de rutas relativas y 28 clases nominales
├── src/
│   ├── detkit/                 # Utilidades comunes: métricas COCO, reportes y resolución portable de rutas
│   ├── cnndet/                 # Detector CNN propio anchor-free tipo CenterNet (CenterDet, 13.13M params)
│   ├── frcnn/                  # Configuración Faster R-CNN ResNet-50 FPN con anclas adaptadas (16..256)
│   ├── explore_dataset.py      # Etapa 1: análisis exploratorio del dataset
│   ├── build_groups.py         # Etapa 1: agrupación perceptual de fotogramas consecutivos (anti-leakage)
│   ├── make_splits.py          # Etapa 1: partición estratificada por grupos (train / val / test)
│   ├── prepare_data.py         # Etapa 1: generación de estructura YOLO en data/fjord/
│   ├── train_cnn.py / eval_cnn.py       # Etapa 2: entrenamiento y evaluación de CNN propia
│   ├── train_yolo.py / eval_yolo.py     # Etapas 3 y 4: YOLO26s desde cero y con Transfer Learning
│   ├── train_frcnn.py / eval_frcnn.py   # Etapa 5: Faster R-CNN con Transfer Learning
│   ├── benchmark_speed.py      # Etapa 6: benchmark uniforme de inferencia (batch=1, fp32)
│   ├── compare_models.py       # Etapa 6: comparación base de métricas y curvas
│   ├── analyze_stage6_extra.py # Etapa 6: tabla unificada, IC 95% Bootstrap, tamaño de objeto y taxonomía de errores
│   └── eval_degradation.py     # Etapa 7: evaluación sistemática bajo 4 tipos de degradación × 3 niveles
├── notebooks/
│   └── Etapa7_Colab_YOLO26s_TL_y_Degradacion.ipynb  # Cuaderno reproducible en GPU (Google Colab)
├── results/
│   ├── eval/                   # Métricas JSON, umbrales óptimos en val y AP por clase
│   ├── preds/                  # Predicciones crudas en val y test para los modelos evaluados
│   ├── comparison/             # Tablas maestras, Bootstrap IC 95%, desglose por escala y taxonomía de errores
│   └── degradation/            # Salidas de Etapa 7 (generadas por eval_degradation.py / Colab)
├── figures/                    # Gráficos de EDA, curvas, comparaciones y mosaicos cualitativos
└── docs/
    ├── Tarea_2-Deep_Learning_2026-2.pdf   # Enunciado oficial
    ├── Rúbrica_T2_DL___2026_2.pdf         # Rúbrica de evaluación
    ├── DECLARACION_IA_PLANTILLA.md        # Plantilla de declaración obligatoria de uso de IA
    └── FICHA_DATOS_PRESENTACION.md        # Tablas verificadas e índice de figuras para las diapositivas
```

---

## Instalación

Se recomienda Python `3.10+` o `3.11`:

```bash
python -m venv .venv
# En Linux / macOS:
source .venv/bin/activate
# En Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Todos los scripts resuelven automáticamente `data/fjord/data.yaml` de manera relativa a la raíz del repositorio (`src/detkit/paths.py`), por lo que funcionan sin modificaciones en Windows, Linux, macOS y Google Colab.

---

## Reproducción por Etapas

### Etapa 1 — Exploración y partición sin fuga temporal

El dataset ya se encuentra particionado en `data/fjord/`. Para regenerar las tablas exploratorias desde los archivos originales:

```bash
python src/explore_dataset.py
python src/build_groups.py
python src/make_splits.py
python src/prepare_data.py
```

### Etapas 2 a 5 — Entrenamiento y evaluación de los 4 modelos

1. **CNN propia desde cero (`cnn_v1`):**
   ```bash
   python src/train_cnn.py --name cnn_v1 --epochs 100 --bs 16 --img-size 640
   python src/eval_cnn.py --name cnn_v1 --split val
   python src/eval_cnn.py --name cnn_v1 --split test
   ```
2. **YOLO26s desde cero (`yolo26s_scratch`):**
   ```bash
   python src/train_yolo.py --mode scratch --model yolo26s --name yolo26s_scratch --epochs 150 --batch 16 --imgsz 640
   python src/eval_yolo.py --name yolo26s_scratch --split val
   python src/eval_yolo.py --name yolo26s_scratch --split test
   ```
3. **YOLO26s con *Transfer Learning* (`yolo26s_tl_frozen` y `yolo26s_tl_full`):**
   ```bash
   python src/train_yolo.py --mode transfer --model yolo26s --name yolo26s_tl_frozen --epochs 50 --batch 16 --freeze 11
   python src/eval_yolo.py --name yolo26s_tl_frozen --split val

   python src/train_yolo.py --mode transfer --model yolo26s --name yolo26s_tl_full --epochs 50 --batch 16 --freeze 0
   python src/eval_yolo.py --name yolo26s_tl_full --split val
   python src/eval_yolo.py --name yolo26s_tl_full --split test
   ```
4. **Faster R-CNN ResNet-50 FPN con *Transfer Learning* (`frcnn_v1`):**
   ```bash
   python src/train_frcnn.py --name frcnn_v1 --epochs 20 --bs 8 --anchors small
   python src/eval_frcnn.py --name frcnn_v1 --split val
   python src/eval_frcnn.py --name frcnn_v1 --split test
   ```

> **Nota sobre checkpoints `.pt`:** En el repositorio están incluidos `runs/cnn/cnn_v1/best.pt` y `runs/yolo/yolo26s_scratch/weights/best.pt`, además de las predicciones completas en `results/preds/*.csv` y métricas en `results/eval/*.json` de los cinco entrenamientos. El checkpoint de `Faster R-CNN` (`>100 MB`) se excluye por límite de tamaño de GitHub, y el checkpoint de `yolo26s_tl_full` puede cargarse o reproducirse directamente en GPU mediante `notebooks/Etapa7_Colab_YOLO26s_TL_y_Degradacion.ipynb`.

### Etapa 6 — Comparación integral, significancia estadística y taxonomía de errores

```bash
# Benchmark uniforme de inferencia en GPU (batch=1, fp32)
python src/benchmark_speed.py --device cuda

# Gráficos comparativos base y análisis estadístico/diagnóstico extendido (a partir de results/preds/)
python src/compare_models.py
python src/analyze_stage6_extra.py
```

Salidas principales en `results/comparison/` y `figures/comparison/`:
- `master_table_unified.md` — Tabla maestra con métricas en `test` y latencia GPU uniforme.
- `bootstrap_ci_test.md` / `bootstrap_paired_diff.csv` / `fig_bootstrap_ci.png` — Intervalos de confianza Bootstrap 95% (`B = 1.000`, pareado por imagen).
- `size_breakdown_test.md` / `fig_map_by_size.png` — `mAP` y `Recall` por tamaño de objeto a escala `640 px` (`<32 px`, `32–96 px`, `>=96 px`).
- `error_taxonomy_test.md` / `top_species_confusions.csv` / `fig_error_taxonomy.png` — Desglose de falsos positivos (confusión de especie, localización, duplicados, fondo) y omisiones (`FN`).

### Etapa 7 — Robustez ante degradación submarina (en local o Google Colab GPU)

Evalúa sin reentrenamiento las `451` imágenes de `test` en memoria bajo **4 familias de degradación × 3 niveles de severidad** (`baja iluminación`, `desenfoque gaussiano`, `ruido gaussiano` y `bajo contraste / turbidez`), manteniendo fijo el umbral óptimo de `val`:

```bash
python src/eval_degradation.py \
  --kind yolo \
  --model-name yolo26s_tl_full \
  --weights runs/yolo/yolo26s_tl_full/weights/best.pt \
  --yaml data/fjord/data.yaml
```

Si no se cuenta con GPU local o con el archivo `runs/yolo/yolo26s_tl_full/weights/best.pt`, abrir y ejecutar de principio a fin el cuaderno **`notebooks/Etapa7_Colab_YOLO26s_TL_y_Degradacion.ipynb`** en Google Colab (`Entorno de ejecución -> Cambiar tipo de entorno -> GPU T4`).

---

## Declaración de Uso de Herramientas de IA

Consultar y completar con los datos de los integrantes la plantilla incluida en [`docs/DECLARACION_IA_PLANTILLA.md`](docs/DECLARACION_IA_PLANTILLA.md) para incorporarla en las diapositivas de entrega.
