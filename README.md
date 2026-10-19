# ✈️ Inspección visual de álabes de motor

Aplicación web que clasifica fotos de álabes de motores de avión en 5 clases (sin defecto, ablación, ruptura, fractura y ranura) usando *transfer learning*, y genera una recomendación de mantenimiento con un modelo de lenguaje.

**App publicada:** _(agregar link de Streamlit Cloud)_

## Flujo

📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Resultado visual → 💬 Gemini Flash → ✅ Recomendación

## Dataset

**AeBAD-S** (*Aero-engine Blade Anomaly Detection*), Zhang et al., *Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction*, arXiv:2304.02216 (2023). Repositorio original: https://github.com/zhangzilongc/MMR

El dataset fue creado para detección de anomalías; se reorganizó como clasificación supervisada de 5 clases con un split propio 70/15/15 estratificado.

## Modelo

Se compararon tres enfoques en el conjunto de prueba:

| Modelo | Accuracy | Acc. balanceada | F1 macro | Defectos pasados como `good` |
|---|---|---|---|---|
| CLIP zero-shot (línea base) | 0.404 | 0.330 | 0.316 | 51 / 172 |
| **EfficientNet-B0** (elegido) | 0.873 | 0.852 | 0.845 | **11 / 172** |
| ResNet-50 | 0.898 | 0.888 | 0.887 | 15 / 172 |

Se eligió **EfficientNet-B0** porque comete menos errores críticos (defectos clasificados como sin defecto), que en inspección aeronáutica es el error más costoso.

## Archivos

- `AeBAD_clasificacion.ipynb`: preparación de datos, entrenamiento, evaluación y exportación del modelo (Google Colab).
- `app.py`: aplicación en Streamlit.
- `modelo_aebad.pt`: modelo entrenado.
- `requirements.txt`: dependencias.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requiere la variable `GEMINI_API_KEY` (en `.streamlit/secrets.toml` o como variable de entorno) para generar recomendaciones.

## Aviso

Proyecto académico con fines educativos. No reemplaza la inspección de un técnico certificado ni el manual del fabricante.
