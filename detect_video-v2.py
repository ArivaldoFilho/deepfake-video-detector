import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import sys
import os
import time
import tempfile

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Detector de Deepfakes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para dar um visual mais "cyberpunk/tech"
st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-weight: bold;
    }
    h2, h3 {
        color: #fafafa;
    }
    .stMetricBox {
        background-color: #262730;
        border: 1px solid #4b4b4b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetricValue {
        font-size: 2rem !important;
        color: #00d4ff !important;
    }
    .stMetricLabel {
        font-size: 1rem !important;
        color: #aaaaaa !important;
    }
    /* Animação pulsante para fakes */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# CARREGAMENTO DE MODELOS (CACHED)
# =====================================================================
MODEL_DIR = "models"
CNN_PATH = os.path.join(MODEL_DIR, "cnn_rvf10k_v2.keras")
DET_MODEL_PATH = os.path.join(MODEL_DIR, "det_logreg_v2.joblib")
STACK_MODEL_PATH = os.path.join(MODEL_DIR, "stacking_v2.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "det_scaler_v2.joblib")
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = (128, 128)


@st.cache_resource
def load_models():
    try:
        if not all(
            os.path.exists(p)
            for p in [CNN_PATH, DET_MODEL_PATH, STACK_MODEL_PATH, SCALER_PATH]
        ):
            st.error(
                "Arquivos de modelo não encontrados na pasta 'models/'. Verifique o treinamento."
            )
            st.stop()

        cnn = tf.keras.models.load_model(CNN_PATH)
        det = joblib.load(DET_MODEL_PATH)
        stack = joblib.load(STACK_MODEL_PATH)
        scl = joblib.load(SCALER_PATH)
        face_casc = cv2.CascadeClassifier(HAAR_PATH)
        return cnn, det, stack, scl, face_casc
    except Exception as e:
        st.error(f"Erro fatal ao carregar modelos: {e}")
        st.stop()


cnn_model, det_model, stack_model, scaler, face_cascade = load_models()


# =====================================================================
# FUNÇÕES AUXILIARES (Extração de Features)
# =====================================================================
def extract_face_bbox(gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_img = cv2.resize(gray[y : y + h, x : x + w], IMG_SIZE)
    return (x, y, w, h), face_img


def lbp_features(img):
    img = img.astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        img = cv2.resize(img, (max(3, w), max(3, h)))
        h, w = img.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = img[i, j]
            code = 0
            code |= (img[i - 1, j - 1] > center) << 7
            code |= (img[i - 1, j] > center) << 6
            code |= (img[i - 1, j + 1] > center) << 5
            code |= (img[i, j + 1] > center) << 4
            code |= (img[i + 1, j + 1] > center) << 3
            code |= (img[i + 1, j] > center) << 2
            code |= (img[i + 1, j - 1] > center) << 1
            code |= (img[i, j - 1] > center) << 0
            lbp[i - 1, j - 1] = code
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist


def extract_deterministic_features(face_gray):
    mean_intensity = np.mean(face_gray)
    std_intensity = np.std(face_gray)
    lap_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    edges = cv2.Canny(face_gray, 100, 200)
    edge_density = np.mean(edges > 0)

    f = np.fft.fft2(face_gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    low = mag[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4].sum()
    total = mag.sum() + 1e-8
    hf_ratio = (total - low) / total

    hist = lbp_features(face_gray)

    features = np.concatenate(
        [
            np.array(
                [mean_intensity, std_intensity, lap_var, edge_density, hf_ratio],
                dtype=np.float32,
            ),
            hist.astype(np.float32),
        ]
    )
    return features.reshape(1, -1)


# =====================================================================
# NÚCLEO DE PROCESSAMENTO DE VÍDEO
# =====================================================================
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    predictions_buffer = []
    frame_probs = []
    frames_with_face = 0

    start_time = time.time()

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processando frame {frame_count}/{total_frames}...")

        if frame_count % 2 != 0:
            out.write(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox, face_gray = extract_face_bbox(gray)

        label_text = "Analisando..."
        color = (100, 100, 100)
        current_prob = 0.5

        if bbox is not None:
            frames_with_face += 1
            x, y, w, h = bbox

            # --- RAMO 1: CNN ---
            face_color = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMG_SIZE)
            cnn_input = np.expand_dims(face_resized, axis=0)
            p_cnn = cnn_model.predict(cnn_input, verbose=0)[0][0]

            # --- RAMO 2: Determinístico ---
            try:
                det_feats = extract_deterministic_features(face_gray)
                det_feats_s = scaler.transform(det_feats)
                p_det = det_model.predict_proba(det_feats_s)[:, 1][0]
            except:
                p_det = 0.5

            # --- RAMO 3: Stacking ---
            stack_input = np.array([[p_cnn, p_det]])
            current_prob = stack_model.predict_proba(stack_input)[:, 1][0]

            # Suavização
            predictions_buffer.append(current_prob)
            if len(predictions_buffer) > 7:
                predictions_buffer.pop(0)

            avg_prob = np.mean(predictions_buffer)
            frame_probs.append(avg_prob)

            # Visualização no frame (desenho para o arquivo processado)
            if avg_prob > 0.5:
                label_text = f"FAKE ({avg_prob:.1%})"
                color = (0, 0, 255)
            else:
                label_text = f"REAL ({1-avg_prob:.1%})"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                frame,
                label_text,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        out.write(frame)

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    end_time = time.time()
    processing_time = end_time - start_time

    # Compilando resultados simplificados
    final_avg_confidence = np.mean(frame_probs) if frame_probs else 0

    results = {
        "avg_confidence": final_avg_confidence,
        "time": processing_time,
        "frames_analyzed": frames_with_face,
    }

    return results


# =====================================================================
# INTERFACE PRINCIPAL
# =====================================================================

st.title("Detector de Deepfakes")
st.markdown("### Sistema Híbrido de Detecção de Manipulação Facial")
st.markdown("---")

# Sidebar simplificado (sem Ground Truth)
with st.sidebar:
    st.header("Configuração da Análise")
    st.write(
        "Faça upload de um vídeo para verificar se ele contém manipulações faciais (Deepfakes)."
    )

    uploaded_file = st.file_uploader(
        "Escolha um arquivo de vídeo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"]
    )

st.write("")  # Espaçamento

# Área principal
if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Vídeo Original")
        st.video(uploaded_file)

    with col2:
        st.subheader("Status")
        if st.button(
            "Iniciar Análise Híbrida", type="primary", use_container_width=True
        ):
            # 1. Salvar o arquivo upado temporariamente
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()  # Libera o arquivo

            # 2. Criar caminho para o vídeo processado
            output_video_path = os.path.join(
                tempfile.gettempdir(), "processed_output.mp4"
            )

            # 3. Processamento com Spinner animado
            with st.spinner(
                "Analisando frames com IA e métodos determinísticos... Aguarde..."
            ):
                try:
                    results = process_video(video_path, output_video_path)
                    st.session_state["results"] = results
                    st.toast("Análise concluída com sucesso!")
                except Exception as e:
                    st.error(f"Ocorreu um erro durante o processamento: {e}")

            # Limpar arquivo temporário de entrada
            try:
                os.remove(video_path)
            except Exception as e:
                pass

# Exibição dos Resultados Simplificados
if "results" in st.session_state:
    st.markdown("---")
    st.header("Resultado Final")

    results = st.session_state["results"]
    final_verdict = "FAKE" if results["avg_confidence"] > 0.5 else "REAL"
    confidence_score = (
        results["avg_confidence"]
        if final_verdict == "FAKE"
        else (1 - results["avg_confidence"])
    )

    # Veredito Principal
    st.markdown(
        f"""
    <div style="text-align: center; padding: 20px; background-color: {'#4a1515' if final_verdict == 'FAKE' else '#154a28'}; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="margin:0;">Veredito Final do Modelo:</h2>
        <h1 style="font-size: 4em; color: {'#ff4b4b' if final_verdict == 'FAKE' else '#4bff85'}; margin: 10px 0;">
            {final_verdict}
        </h1>
        <h3>Confiança Média Global: {confidence_score:.1%}</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Estatísticas de processamento apenas
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Tempo de Processamento", f"{results['time']:.2f} segundos")
    with m_col2:
        st.metric("Frames Analisados (com rosto)", f"{results['frames_analyzed']}")

else:
    # Estado inicial (placeholder)
    st.info("Faça o upload de um vídeo na barra lateral para começar.")
