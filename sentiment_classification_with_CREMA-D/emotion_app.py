
import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Speech Emotion Recognizer", page_icon="🎭", layout="centered")

MAX_RECORD_SECONDS = 10   # Max buffer; user stops whenever they want

@st.cache_resource
def load_artifacts():
    return joblib.load("emotion_artifacts.pkl")

artifacts        = load_artifacts()
final_models     = artifacts["final_models"]
scaler           = artifacts["scaler"]
TARGET_SAMPLES   = artifacts["TARGET_SAMPLES"]
SR               = artifacts["SR"]
label_to_emotion = artifacts["label_to_emotion"]

EMOTION_EMOJI = {
    "Anger": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy/Joy": "😊", "Neutral": "😐", "Sad": "😢"
}

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("is_recording", False), ("frames", None), ("result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎭 Speech Emotion Recognizer")
st.markdown("Press **Record**, speak naturally, then press **Stop** to get the prediction.")
st.divider()

# ── Record / Stop button ──────────────────────────────────────────────────────
_, col, _ = st.columns([1, 2, 1])

with col:
    if not st.session_state.is_recording:
        # Idle state: show "Record" or "Record New" (if there is a previous result)
        label = "⏺  Record New" if st.session_state.result else "⏺  Record"
        if st.button(label, use_container_width=True, type="primary"):
            st.session_state.result   = None       # Clear previous result
            st.session_state.frames   = sd.rec(    # Start buffered recording
                int(MAX_RECORD_SECONDS * SR),
                samplerate = SR,
                channels   = 1,
                dtype      = "float32"
            )
            st.session_state.is_recording = True
            st.rerun()
    else:
        # Recording state: animate indicator + show Stop button
        st.markdown(
            "<div style='text-align:center; padding:8px; border-radius:8px;"
            "background:#fdecea; color:#c0392b; font-size:1.15rem;'>"
            "🔴 &nbsp;<b>Recording...</b></div>",
            unsafe_allow_html=True
        )
        st.write("")
        if st.button("⏹  Stop & Predict", use_container_width=True, type="primary"):
            sd.stop()                               # Stop sounddevice stream

            # Trim trailing silence (pre-allocated buffer is filled with zeros)
            raw = st.session_state.frames.flatten()
            nonzero = np.nonzero(raw)[0]
            y = raw[: nonzero[-1] + 1] if len(nonzero) > 0 else raw

            # ── Same pipeline as training ─────────────────────────────────────
            # Step 1: Standardize length
            if len(y) > TARGET_SAMPLES:
                y = y[:TARGET_SAMPLES]
            else:
                y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")

            # Step 2: Extract RMS
            rms   = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
            X_new = rms[0].reshape(1, -1)

            # Step 3: Apply scaler fitted on training data
            X_new_scaled = scaler.transform(X_new)

            # Step 4: Predict with both models
            predictions = {}
            for name, model in final_models.items():
                pred_label         = model.predict(X_new_scaled)[0]
                predictions[name]  = label_to_emotion[pred_label]

            # Save result to session state
            st.session_state.result = {
                "predictions": predictions,
                "y"          : y,
                "X_new"      : X_new,
            }
            st.session_state.is_recording = False
            st.session_state.frames       = None
            st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result
    st.divider()
    st.subheader("🎯 Predicted Emotion")

    for col, (name, emotion) in zip(st.columns(len(res["predictions"])), res["predictions"].items()):
        with col:
            emoji = EMOTION_EMOJI.get(emotion, "🎭")
            st.metric(label=name, value=f"{emoji} {emotion}")

    st.divider()
    st.subheader("📊 Audio Analysis")

    y     = res["y"]
    X_new = res["X_new"]
    time_axis = np.linspace(0, len(y) / SR, len(y))
    rms_time  = librosa.frames_to_time(np.arange(X_new.shape[1]), sr=SR, hop_length=512)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].plot(time_axis, y, color="#1f77b4", linewidth=0.7)
    axes[0].set_title("Waveform (standardized)")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(rms_time, X_new[0], color="#e74c3c", linewidth=1.2)
    axes[1].set_title("RMS Energy (extracted features)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)
