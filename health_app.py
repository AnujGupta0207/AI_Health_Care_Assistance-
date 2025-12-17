# health_app.py (complete improved UI)
import os
import glob
import json
import pickle
import random
import io
import datetime
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# your symptom engine (keeps previous behavior)
from symptom_engine import predict_diseases

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stemmer = LancasterStemmer()

# Set page config with improved theme
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (existing from your code)
st.markdown("""<style>
/* (trimmed for brevity in the message; keep your full CSS from previous code) */
</style>""", unsafe_allow_html=True)

# -------------------- Helpers: model loading & mapping --------------------
@st.cache_resource
def load_ecg_model_and_mapping():
    model_path = os.path.join(BASE_DIR, "ecg_model.h5")
    if not os.path.exists(model_path):
        return None, None

    model = tf.keras.models.load_model(model_path, compile=False)

    # auto-detect mapping (P(normal) or P(arrhythmia))
    sample_dir = os.path.join(BASE_DIR, "data", "ecg_images")
    mapping = None
    if os.path.isdir(sample_dir):
        # support multiple extensions and case variations
        patterns = ["*normal*.png", "*normal*.jpg", "*normal*.jpeg", "*normal*.PNG", "*normal*.JPG", "*normal*.JPEG"]
        normal_paths = []
        for p in patterns:
            normal_paths.extend(glob.glob(os.path.join(sample_dir, p)))
        patterns2 = ["*arrhythmia*.png", "*arrhythmia*.jpg", "*arrhythmia*.jpeg", "*arrhythmia*.PNG", "*arrhythmia*.JPG", "*arrhythmia*.JPEG"]
        arr_paths = []
        for p in patterns2:
            arr_paths.extend(glob.glob(os.path.join(sample_dir, p)))

        normal_paths = sorted(list(dict.fromkeys(normal_paths)))[:10]
        arr_paths = sorted(list(dict.fromkeys(arr_paths)))[:10]

        if normal_paths and arr_paths:
            def raw_pred(p):
                img = load_img(p, color_mode="grayscale", target_size=(128,128))
                arr = img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                return float(model.predict(arr)[0][0])

            avg_norm = float(np.mean([raw_pred(p) for p in normal_paths]))
            avg_arr = float(np.mean([raw_pred(p) for p in arr_paths]))

            mapping = {"pred_is": "P(normal)" if avg_norm > avg_arr else "P(arrhythmia)",
                       "avg_normal": avg_norm, "avg_arr": avg_arr}
    return model, mapping

@st.cache_resource
def load_chatbot():
    intents = {}
    words = []
    labels = []
    chat_model = None
    intents_path = os.path.join(BASE_DIR, "intents.json")
    data_pickle = os.path.join(BASE_DIR, "data.pickle")
    chat_model_path = os.path.join(BASE_DIR, "model.h5")

    if os.path.exists(intents_path):
        with open(intents_path, "r") as f:
            intents = json.load(f)
    if os.path.exists(data_pickle):
        with open(data_pickle, "rb") as f:
            words, labels, training, output = pickle.load(f)
    if os.path.exists(chat_model_path):
        chat_model = tf.keras.models.load_model(chat_model_path, compile=False)
    return intents, words, labels, chat_model

ecg_model, ecg_mapping = load_ecg_model_and_mapping()
intents, words, labels, chat_model = load_chatbot()

# -------------------- Utilities --------------------
def bag_of_words(sentence, words_list):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(w.lower()) for w in sentence_words]
    bag = [0] * len(words_list)
    for s in sentence_words:
        for i, w in enumerate(words_list):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def chatbot_response(text):
    text = text.strip()
    if not text:
        return "Please write something."

    if "," in text:
        possible = predict_diseases(text)
        if not possible:
            return "Couldn't match symptoms clearly. Please consult a doctor."
        msg = "Possible conditions (based on symptom matching):\n"
        for disease, score in possible:
            msg += f"- {disease.replace('_',' ').title()} (matched {score} symptom(s))\n"
        msg += "\n⚠ This is not a confirmed diagnosis. See a doctor."
        return msg

    if chat_model is None:
        return "Chat model not available. Try symptom input with commas."

    bow = bag_of_words(text, words)
    res = chat_model.predict(np.array([bow]))[0]
    idx = int(np.argmax(res))
    tag = labels[idx] if labels else None
    if res[idx] > 0.5 and tag:
        for tg in intents.get("intents", []):
            if tg["tag"] == tag:
                return random.choice(tg["responses"])
    return "I didn't understand. Try listing your symptoms separated by commas."

def predict_raw_ecg(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(128,128))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return float(ecg_model.predict(arr)[0][0])

def interpret_raw_ecg(pred, mapping, manual_mode=None):
    # manual_mode: None / "P(normal)" / "P(arrhythmia)"
    if manual_mode is not None:
        mapping_use = {"pred_is": manual_mode}
    else:
        mapping_use = mapping

    if mapping_use is None:
        # fallback assume P(normal)
        p_normal = pred
        p_arr = 1.0 - pred
    else:
        if mapping_use["pred_is"] == "P(normal)":
            p_normal = pred
            p_arr = 1.0 - pred
        else:
            p_arr = pred
            p_normal = 1.0 - pred

    label = "ARRHYTHMIA" if p_arr >= 0.5 else "NORMAL"
    return label, p_arr, p_normal

def plot_waveform_from_image(pil_img):
    # Convert PIL image to grayscale numpy, then plot the central row as a proxy waveform.
    gray = pil_img.convert("L")
    arr = np.asarray(gray).astype(float)
    h, w = arr.shape
    row = arr[int(h * 0.5)]
    row = (row - row.mean()) / (np.abs(row).max() + 1e-6)
    fig, ax = plt.subplots(figsize=(6,1.6))
    ax.plot(row, linewidth=1, color='#4ECDC4')
    ax.axis("off")
    plt.tight_layout()
    return fig

# -------------------- Session: history and user profile --------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"name": "", "age": "", "gender": ""}
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# Notification helpers
def add_notification(message, type="info"):
    st.session_state.notifications.append({"message": message, "type": type, "time": time.time()})

def show_notifications():
    for notif in st.session_state.notifications[:]:
        # show only recent ones
        if time.time() - notif["time"] > 6:
            st.session_state.notifications.remove(notif)
        else:
            color = "#4CAF50" if notif["type"]=="success" else "#2196F3" if notif["type"]=="info" else "#FF9800"
            st.markdown(f'<div style="position:fixed;top:20px;right:20px;background:{color};color:white;padding:10px 18px;border-radius:8px;z-index:9999">{notif["message"]}</div>', unsafe_allow_html=True)

show_notifications()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("🏥 AI Healthcare Assistant")
    st.markdown("Demo app: symptom chat + ECG analyzer")
    with st.expander("👤 User Profile"):
        st.session_state.user_profile["name"] = st.text_input("Name", value=st.session_state.user_profile["name"])
        st.session_state.user_profile["age"] = st.text_input("Age", value=st.session_state.user_profile["age"])
        st.session_state.user_profile["gender"] = st.selectbox("Gender", ["", "Male", "Female", "Other"], index=0 if not st.session_state.user_profile["gender"] else ["", "Male", "Female", "Other"].index(st.session_state.user_profile["gender"]))
    st.markdown("---")
    st.subheader("Model Status")
    if ecg_model is None:
        st.error("ECG model not found (`ecg_model.h5`).")
    else:
        if ecg_mapping:
            st.write(f"Auto-detected: **{ecg_mapping['pred_is']}**")
            st.caption(f"avg_normal={ecg_mapping['avg_normal']:.3f} · avg_arr={ecg_mapping['avg_arr']:.3f}")
        else:
            st.info("Could not auto-detect mapping. Using fallback: P(normal).")
    st.markdown("---")
    st.subheader("Quick Actions")
    if st.button("🔄 Re-run Mapping"):
        st.experimental_rerun()
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        add_notification("History cleared!", "success")
    if st.button("📤 Export History"):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "history.csv", "text/csv")
        else:
            st.warning("No history to export.")
    st.markdown("---")
    st.caption("Built as an educational demo. Not a medical device.")

# -------------------- Main layout --------------------
st.title("🏥 AI Healthcare Assistant")
st.write("Chat with the assistant or analyze an ECG image. This is a demo — not medical advice.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["💬 Symptom Chat", "❤️ ECG Analyzer", "📊 Dashboard"])

# -------------------- Tab 1: Chat --------------------
with tab1:
    st.subheader("Symptom Chat & Assistant")
    left, right = st.columns([3,1])
    with left:
        user_text = st.text_area("Describe symptoms or ask a question", height=120, placeholder="e.g., fever, headache, body pain")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Ask Assistant"):
                if user_text.strip():
                    with st.spinner("Thinking..."):
                        resp = chatbot_response(user_text)
                    st.markdown("**🤖 Assistant:**")
                    st.write(resp)
                    st.session_state.history.insert(0, {"type":"chat", "input": user_text, "output": resp, "time": str(datetime.datetime.now())})
                    add_notification("Response generated!", "success")
                else:
                    st.warning("Type a message first.")
        with col2:
            feedback = st.radio("Rate this response:", ["", "👍 Good", "👎 Bad"], key="chat_feedback")
            if feedback:
                add_notification("Thank you for your feedback!", "info")
    with right:
        st.markdown("### 💡 Tips")
        st.write("- For symptoms, list them with commas: `fever, cough, body pain`")
        st.write("- For general questions, just ask normally")
        st.write("- This assistant suggests possibilities — not diagnosis")

# -------------------- Tab 2: ECG Analyzer --------------------
with tab2:
    st.subheader("ECG Analyzer (single-beat / single-window images)")
    c1, c2 = st.columns([2,1])
    with c1:
        uploaded_file = st.file_uploader("Upload ECG image (PNG/JPG)", type=["png","jpg","jpeg"])
        sample_dir = os.path.join(BASE_DIR, "data", "ecg_images")
        # discover samples robustly
        patterns = ["*.png","*.PNG","*.jpg","*.JPG","*.jpeg","*.JPEG"]
        samples = []
        if os.path.isdir(sample_dir):
            for patt in patterns:
                samples.extend(glob.glob(os.path.join(sample_dir, patt)))
        samples = sorted(list(dict.fromkeys(samples)))
        sample_names = [os.path.basename(p) for p in samples]
        if not samples:
            st.warning("No sample images found in data/ecg_images. Run ecg_data_prep.py to generate them.")
            sample_choice = st.selectbox("Choose sample image (optional)", options=[""])
        else:
            sample_choice = st.selectbox("Choose sample image (optional)", options=[""] + sample_names)

        pil = None
        selected_path = None
        # if user selected sample, load it
        if sample_choice:
            selected_path = os.path.join(sample_dir, sample_choice)
            if os.path.exists(selected_path):
                try:
                    pil = Image.open(selected_path).convert("RGB")
                except Exception as e:
                    st.error(f"Could not open sample image: {e}")
                    pil = None

        # if user uploaded, use that (overrides sample)
        if uploaded_file is not None:
            try:
                pil = Image.open(uploaded_file)
            except Exception as e:
                st.error("Uploaded file could not be read: " + str(e))
                pil = None

        manual_mode = st.radio("Mapping override (if auto-detected mapping appears wrong):", options=["Auto-detect", "Force P(normal)", "Force P(arrhythmia)"])
        manual_map = None
        if manual_mode == "Force P(normal)":
            manual_map = "P(normal)"
        elif manual_mode == "Force P(arrhythmia)":
            manual_map = "P(arrhythmia)"

        analyze_btn = st.button("🔍 Analyze ECG")

    with c2:
        st.markdown("### Model info")
        if ecg_model is None:
            st.error("ECG model not found.")
        else:
            st.write("Model loaded ✅")
            if ecg_mapping:
                st.info(f"Auto mapping: **{ecg_mapping['pred_is']}**  (avg_norm={ecg_mapping['avg_normal']:.3f}, avg_arr={ecg_mapping['avg_arr']:.3f})")
            else:
                st.info("Mapping: fallback (assume P(normal))")
            st.markdown("**Note:** This is a demo model; use images similar to training set for best results.")

    if analyze_btn:
        if pil is None:
            st.warning("Please upload an image or pick a sample first.")
        elif ecg_model is None:
            st.error("No ECG model available to run prediction.")
        else:
            # save temp file
            tmp_path = os.path.join(BASE_DIR, "temp_ecg_upload.png")
            try:
                pil.save(tmp_path)
            except Exception as e:
                st.error("Could not save temporary image: " + str(e))
                tmp_path = None

            if tmp_path:
                with st.spinner("Running model inference..."):
                    try:
                        raw = predict_raw_ecg(tmp_path)
                        label, p_arr, p_norm = interpret_raw_ecg(raw, ecg_mapping, manual_map)
                    except Exception as e:
                        st.error("Model inference failed: " + str(e))
                        raw = None
                        label = None
                        p_arr = None
                        p_norm = None

                if raw is not None:
                    st.image(pil, caption="Input ECG image", use_column_width=True)
                    fig = plot_waveform_from_image(pil)
                    st.pyplot(fig)

                    st.markdown("### Result")
                    st.metric("Final classification", label)
                    col_a, col_b = st.columns([3,1])
                    with col_a:
                        st.write(f"**P(arrhythmia):** {p_arr:.4f}  •  **P(normal):** {p_norm:.4f}")
                        # progress bar shows arrhythmia probability
                        st.progress(min(max(p_arr, 0.0), 1.0))
                    with col_b:
                        if p_arr >= 0.5:
                            st.markdown("<div style='background:#ffdddd;padding:10px;border-radius:8px'><b style='color:#d32f2f'>High arrhythmia probability</b></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='background:#ddffeb;padding:10px;border-radius:8px'><b style='color:#2e7d32'>Low arrhythmia probability</b></div>", unsafe_allow_html=True)

                    # save to history
                    rec = {"type":"ecg", "file": os.path.basename(tmp_path), "label": label,
                           "p_arr": round(p_arr,4), "p_norm": round(p_norm,4), "time": str(datetime.datetime.now()),
                           "user": st.session_state.user_profile.get("name","")}
                    st.session_state.history.insert(0, rec)

                    # downloadable report
                    report_txt = f"""AI ECG Report
Time: {rec['time']}
User: {rec['user']}
File: {rec['file']}
Prediction: {rec['label']}
P(arrhythmia): {rec['p_arr']}
P(normal): {rec['p_norm']}

Disclaimer: This is a demo model. Not a medical diagnosis.
"""
                    st.download_button("📥 Download text report", report_txt, file_name=f"ecg_report_{rec['time'].replace(':','-')}.txt")
                    add_notification("ECG analyzed and saved to history.", "success")

# -------------------- Tab 3: Dashboard --------------------
with tab3:
    st.subheader("Session Dashboard")
    if not st.session_state.history:
        st.info("No activity recorded this session yet. Run a chat or ECG analysis to populate the dashboard.")
    else:
        df = pd.DataFrame(st.session_state.history)
        # Show recent activity table
        st.markdown("### Recent activity")
        st.dataframe(df.head(20))

        # ECG-only stats
        ecg_df = df[df["type"]=="ecg"] if "type" in df.columns else pd.DataFrame()
        if not ecg_df.empty:
            st.markdown("### ECG predictions summary")
            # counts by label
            counts = ecg_df["label"].value_counts().reset_index()
            counts.columns = ["label","count"]
            fig1 = px.pie(counts, names="label", values="count", title="Prediction distribution")
            st.plotly_chart(fig1, use_container_width=True)

            # P(arrhythmia) over time
            ecg_df["time_parsed"] = pd.to_datetime(ecg_df["time"])
            fig2 = px.line(ecg_df.sort_values("time_parsed"), x="time_parsed", y="p_arr", markers=True, title="P(arrhythmia) over time")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Sample raw table")
            st.dataframe(ecg_df[["time","file","label","p_arr","p_norm","user"]].head(50))
        else:
            st.info("No ECG analyses yet.")

# -------------------- End --------------------
st.markdown("---")
st.caption("This app is for educational/demo purposes only. Not for clinical use.")

        
