"""
App: Inspección visual de álabes de motores de avión (AeBAD)
Flujo: Imagen -> Modelo preentrenado (clase + confianza) -> Resultado visual -> LLM -> Recomendación
"""
import os
import torch
import timm
import streamlit as st
from PIL import Image
from torchvision import transforms
import anthropic

st.set_page_config(page_title="Inspección de álabes", page_icon="✈️", layout="wide")

NOMBRES_ES = {
    "good": "Sin defecto",
    "ablation": "Ablación (quemadura / desgaste térmico)",
    "breakdown": "Ruptura (pérdida de material)",
    "fracture": "Fractura (grieta)",
    "groove": "Ranura (rayado superficial)",
}

# Umbrales de confianza
UMBRAL_GENERAL = 0.60  # bajo esto, cualquier resultado requiere revisión humana
UMBRAL_GOOD = 0.80     # para declarar "sin defecto" se exige más confianza (error más costoso)

# Métricas del conjunto de PRUEBA (secciones 7 y 8 del notebook)
METRICAS_TEST = {
    "Accuracy": 0.873,
    "Accuracy balanceada": 0.852,
    "F1 macro": 0.845,
    "Recall de defecto": 0.936,
}

PROMPT_SISTEMA = """Eres un asistente de apoyo para técnicos de mantenimiento aeronáutico.
Recibes el resultado de un modelo de visión que clasifica fotos de álabes de motores de avión.

Reglas:
- Responde en español, en máximo 120 palabras, con 3 secciones en negrita: "Qué significa", "Acción sugerida", "Precaución".
- Menciona siempre el nivel de confianza del modelo.
- Si se indica que el resultado requiere revisión humana, NO des un diagnóstico: recomienda inspección manual por un técnico certificado.
- No inventes medidas, límites del manual ni números de parte. No conoces el motor específico.
- Deja claro que es una herramienta de apoyo y que la decisión final corresponde a un inspector certificado según el manual del fabricante.
- Si el resultado es "sin defecto", recuerda que el modelo puede fallar y que no reemplaza la inspección reglamentaria."""


# ---------- 1. Cargar el modelo (una sola vez) ----------
@st.cache_resource
def cargar_modelo():
    paquete = torch.load("modelo_aebad.pt", map_location="cpu", weights_only=False)
    modelo = timm.create_model(paquete["arquitectura"], pretrained=False,
                               num_classes=len(paquete["clases"]))
    modelo.load_state_dict(paquete["state_dict"])
    modelo.eval()
    tf = transforms.Compose([
        transforms.Resize((paquete["img_size"], paquete["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(paquete["media"], paquete["std"]),
    ])
    return modelo, tf, paquete["clases"], paquete["arquitectura"]


modelo, tf, CLASES, ARQUITECTURA = cargar_modelo()


def requiere_revision(clase, confianza):
    if confianza < UMBRAL_GENERAL:
        return True
    if clase == "good" and confianza < UMBRAL_GOOD:
        return True
    return False


def clasificar(imagen):
    x = tf(imagen.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0]
    resultados = {}
    for i in range(len(CLASES)):
        resultados[CLASES[i]] = float(probs[i])
    idx = int(probs.argmax())
    return CLASES[idx], float(probs[idx]), resultados


# ---------- 2. LLM ----------
def obtener_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def recomendar(clase, confianza, resultados, revision):
    orden = sorted(resultados.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top3 = ", ".join(f"{NOMBRES_ES[c]} {p:.0%}" for c, p in orden)
    mensaje = (
        f"Resultado del modelo: {NOMBRES_ES[clase]} (clase '{clase}').\n"
        f"Confianza: {confianza:.0%}.\n"
        f"Otras opciones del modelo: {top3}.\n"
        f"¿Requiere revisión humana por baja confianza?: {'SÍ' if revision else 'NO'}.\n"
        "Genera la recomendación."
    )
    api_key = obtener_api_key()
    if not api_key:
        return "Falta configurar la API key del LLM. Consulte a un inspector certificado."
    try:
        cliente = anthropic.Anthropic(api_key=api_key)
        respuesta = cliente.messages.create(
            model="claude-haiku-5-5",
            max_tokens=400,
            system=PROMPT_SISTEMA,
            messages=[{"role": "user", "content": mensaje}],
        )
        return respuesta.content[0].text
    except Exception:
        return "No se pudo generar la recomendación automática. Consulte a un inspector certificado."


# ---------- 3. Interfaz ----------
st.title("✈️ Inspección visual de álabes de motor")
st.write("Sube una foto de un álabe. El modelo detecta si tiene un defecto y de qué tipo, "
         "y un asistente de IA sugiere los próximos pasos.")
st.info("**Uso educativo.** Esta herramienta es un proyecto académico de apoyo. "
        "No reemplaza la inspección de un técnico certificado ni el manual del fabricante.", icon="ℹ️")

with st.sidebar:
    st.header("📊 Desempeño del modelo")
    st.caption(f"Modelo: **{ARQUITECTURA}**, evaluado en el conjunto de prueba "
               "(15% de las imágenes, no vistas en el entrenamiento).")
    for nombre in METRICAS_TEST:
        st.metric(nombre, f"{METRICAS_TEST[nombre]:.1%}")
    st.caption("Estas métricas describen al modelo en general. "
               "La confianza que aparece al analizar es solo para esa imagen.")

col_izq, col_der = st.columns(2, gap="large")

with col_izq:
    st.subheader("1. Imagen")
    archivo = st.file_uploader("Foto del álabe", type=["png", "jpg", "jpeg"])

    ejemplo = None
    if os.path.isdir("ejemplos"):
        archivos_ejemplo = sorted(f for f in os.listdir("ejemplos") if not f.startswith("."))
        if archivos_ejemplo:
            ejemplo = st.selectbox("…o elige una imagen de ejemplo",
                                   ["(ninguna)"] + archivos_ejemplo)
            if ejemplo == "(ninguna)":
                ejemplo = None

    imagen = None
    if archivo is not None:
        imagen = Image.open(archivo)
    elif ejemplo is not None:
        imagen = Image.open(os.path.join("ejemplos", ejemplo))

    if imagen is not None:
        st.image(imagen, width="stretch")
        if st.button("Analizar imagen", type="primary", width="stretch"):
            clase, confianza, resultados = clasificar(imagen)
            st.session_state["resultado"] = {
                "clase": clase, "confianza": confianza, "resultados": resultados,
                "revision": requiere_revision(clase, confianza),
            }
            st.session_state.pop("recomendacion", None)

with col_der:
    st.subheader("2. Resultado")
    r = st.session_state.get("resultado")
    if r is None:
        st.caption("Aquí aparecerá la clasificación cuando analices una imagen.")
    else:
        st.markdown(f"### {NOMBRES_ES[r['clase']]}")
        st.write(f"Confianza en esta imagen: **{r['confianza']:.0%}**")

        if r["revision"]:
            st.warning("**Revisión humana recomendada.** La confianza del modelo no es suficiente "
                       "para un resultado concluyente.", icon="⚠️")

        st.caption("Probabilidad por clase")
        orden = sorted(r["resultados"].items(), key=lambda kv: kv[1], reverse=True)
        for c, p in orden:
            st.progress(p, text=f"{NOMBRES_ES[c]}: {p:.1%}")

        st.subheader("3. Recomendación")
        if st.button("Generar recomendación con IA", width="stretch"):
            with st.spinner("Consultando al asistente…"):
                st.session_state["recomendacion"] = recomendar(
                    r["clase"], r["confianza"], r["resultados"], r["revision"])
        if "recomendacion" in st.session_state:
            st.markdown(st.session_state["recomendacion"])
