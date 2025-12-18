import streamlit as st
from gradio_client import Client
import streamlit.components.v1 as components
import base64
import os
import time

# ================= CONFIG =================
st.set_page_config(
    page_title="Text to Speech",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}
#MainMenu, footer, header {visibility: hidden;}

.hero-title {
    font-size: 52px;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
}
.hero-subtitle {
    font-size: 20px;
    color: #cbd5f5;
    text-align: center;
    margin-bottom: 40px;
}
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 32px;
    border-radius: 22px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    margin-bottom: 30px;
}
.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 16px;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    color: white;
    border-radius: 14px;
    padding: 12px 26px;
    font-size: 16px;
    font-weight: 700;
    border: none;
}
.profile-card {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    color: white;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# ================= AUDIO PLAYER WITH SPEED =================
def audio_player_with_speed(audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode()

    html = f"""
    <div style="background:rgba(255,255,255,0.08);
                padding:20px;
                border-radius:16px;">
        <audio id="audio" controls style="width:100%;">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>

        <p style="color:white;margin-top:10px;">
            Playback Speed:
            <input type="range" min="0.5" max="2.0" step="0.25"
                   value="1.0"
                   oninput="document.getElementById('audio').playbackRate = this.value">
            <span style="color:#93c5fd;">0.5x – 2.0x</span>
        </p>
    </div>
    """
    components.html(html, height=160)

# ================= HERO =================
st.markdown('<div class="hero-title">🔊 VITS Neural Text-to-Speech Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Natural Voice Generation powered by VITS & Deep Learning</div>', unsafe_allow_html=True)

# ================= TABS =================
tab_home, tab_demo, tab_about, tab_profile = st.tabs(
    ["🏠 Overview", "🎧 Try the Demo", "📘 Technology", "👥 Team"]
)

# ================= DEMO =================
with tab_demo:
    st.markdown('<div class="section-title">🎧 Text to Speech Demo</div>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1.1, 1])

    with col_input:
        st.markdown("#### 📝 Text Input")
        text_input = st.text_area(
            "Text Input",
            placeholder="Hello, our beloved Bahlil!",
            height=180,
            label_visibility="collapsed"
        )
        generate = st.button("🔊 Generate Voice")

    if generate:
        if not text_input.strip():
            st.warning("⚠️ Text input cannot be empty.")
        else:
            progress = st.progress(0)
            for i in range(0, 101, 20):
                time.sleep(0.1)
                progress.progress(i)

            try:
                client = Client("rfauzannn07/tts_project")
                result = client.predict(
                    text=text_input,
                    api_name="/synthesize"
                )
                if os.path.exists(result):
                    st.session_state["audio_path"] = result
                    st.success("✅ Voice generated successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col_output:
        st.markdown("#### 🔊 Audio Output")
        if "audio_path" in st.session_state:
            audio_player_with_speed(st.session_state["audio_path"])

            with open(st.session_state["audio_path"], "rb") as f:
                st.download_button(
                    "⬇️ Download WAV",
                    f,
                    file_name="tts_output.wav",
                    mime="audio/wav"
                )
        else:
            st.info("🎧 Audio output akan muncul setelah proses generate.")

# ================= PROFILE =================
with tab_profile:
    st.markdown('<div class="section-title">👥 Our Team</div>', unsafe_allow_html=True)
    members = [
        ("Refa Muhammad", "1227050113"),
        ("Yusuf Ginanjar", "1227050136"),
        ("Rizkco Fauzan Adhim", "1227050117"),
        ("Onixa Shafa Putri Wibowo", "1227050107"),
    ]
    cols = st.columns(4)
    for col, (name, nim) in zip(cols, members):
        with col:
            st.markdown(f"""
            <div class="profile-card">
                <b>{name}</b><br/>
                <small>{nim}</small>
            </div>
            """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    NLP Project • Text to Speech • Streamlit × Hugging Face
</div>
""", unsafe_allow_html=True)
