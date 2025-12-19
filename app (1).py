import streamlit as st
import streamlit.components.v1 as components
from gradio_client import Client
import io
import os
import time
import base64
import numpy as np
import librosa
import matplotlib.pyplot as plt

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
/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero */
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

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 32px;
    border-radius: 22px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    margin-bottom: 30px;
}

/* Section Title */
.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 16px;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    color: white;
    border-radius: 14px;
    padding: 12px 26px;
    font-size: 16px;
    font-weight: 700;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
}

/* Profile Card */
.profile-card {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    color: white;
    transition: 0.3s;
}

.profile-card:hover {
    background: rgba(255,255,255,0.15);
    transform: translateY(-6px);
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

def change_speed(audio_path, speed=1.0):
    audio = AudioSegment.from_file(audio_path, format="wav")
    new_audio = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)

    output_path = f"temp_speed_{speed}.wav"
    new_audio.export(output_path, format="wav")

    return output_path

# ================= HERO =================
st.markdown('<div class="hero-title">🔊 VITS Neural Text-to-Speech Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Natural Voice Generation powered by VITS & Deep Learning</div>',
    unsafe_allow_html=True
)

# ================= TABS =================
tab_home, tab_demo, tab_about, tab_profile = st.tabs(
    ["🏠 Overview", "🎧 Try the Demo", "📘 Technology", "👥 Team"]
)

# ================= HOME =================
with tab_home:
    st.markdown('<div class="section-title">❓ Overview</div>', unsafe_allow_html=True)

    # ---- INTRO CARD ----
    st.markdown("""
    <div class="glass-card">
        <h3>🔊 What is Text-to-Speech?</h3>
        <p>
        Dashboard ini merupakan <b>aplikasi Text-to-Speech (TTS)</b> berbasis 
        <b>deep learning</b> yang mampu mengonversi teks tertulis menjadi suara manusia
        secara <b>otomatis, natural, dan end-to-end</b>.
        </p>
        <p>
        Sistem ini dikembangkan sebagai <b>demo interaktif</b> untuk menunjukkan
        bagaimana model TTS modern dapat diintegrasikan ke dalam aplikasi web.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- 3 COLUMN CARDS ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>⚙️ How It Works</h3>
            <ol>
                <li>User memasukkan teks</li>
                <li>Teks dikirim ke model TTS</li>
                <li>Model menghasilkan audio</li>
                <li>Audio diputar di dashboard</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🧠 Model</h3>
            <p>
            Sistem ini menggunakan <b>VITS</b> 
            (Variational Inference with Adversarial Learning for
            End-to-End Text-to-Speech).
            </p>
            <ul>
                <li>End-to-end architecture</li>
                <li>Natural voice output</li>
                <li>Modern deep learning TTS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Dataset</h3>
            <p>
            Model dilatih menggunakan <b>Jenny TTS Dataset</b>, 
            dataset open-source berisi pasangan teks dan audio
            yang telah melalui proses preprocessing dan text cleaning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- GOAL CARD ----
    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Project Goals</h3>
        <ul>
            <li>Menyediakan demo TTS berbasis AI</li>
            <li>Menunjukkan integrasi model & web app</li>
            <li>Implementasi NLP & Speech Processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info("👉 Silakan lanjut ke tab **Try the Demo** untuk mencoba konversi teks ke suara secara langsung.")

# ================= DEMO =================
with tab_demo:
    st.markdown('<div class="section-title">🎧 Text to Speech Demo</div>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1.15, 1])

    # ================= LEFT PANEL =================
    with col_input:
        st.markdown("#### 📝 Text Input")
        st.caption("Masukkan teks Bahasa Inggris yang ingin diubah menjadi suara")

        text_input = st.text_area(
            "Text Input",
            placeholder="Hello, everyone this is TTS!",
            height=180,
            label_visibility="collapsed"
        )

        char_count = len(text_input.strip())
        st.caption(f"🔢 Character count: **{char_count}**")

        generate = st.button("🔊 Generate Voice", use_container_width=True)

    # ================= GENERATION PROCESS =================
    if generate:
        if not text_input.strip():
            st.warning("⚠️ Text input cannot be empty.")
        else:
            status = st.empty()
            progress = st.progress(0)

            start_time = time.time()

            for i in range(0, 101, 20):
                time.sleep(0.1)
                progress.progress(i)

            try:
                status.info("🧠 Synthesizing voice using VITS model...")

                client = Client("rfauzannn07/tts_project")
                result = client.predict(
                    text=text_input,
                    api_name="/synthesize"
                )

                if os.path.exists(result):
                    with open(result, "rb") as f:
                        audio_bytes = f.read()

                    latency = round(time.time() - start_time, 2)

                    st.session_state["audio_bytes"] = audio_bytes
                    st.session_state["latency"] = latency
                    st.session_state["tts_success"] = True

            except Exception as e:
                status.error(f"❌ Error: {e}")

    # ================= SUCCESS MESSAGE =================
    if st.session_state.get("tts_success"):
        st.success("✅ Voice generated successfully!")

    # ================= RIGHT PANEL =================
    with col_output:
        st.markdown("#### 🔊 Audio Output")
        st.caption("Preview, adjust speed, and download the generated audio")

        if "audio_bytes" in st.session_state:
            audio_base64 = base64.b64encode(
                st.session_state["audio_bytes"]
            ).decode()

            # ================= AUDIO PLAYER =================
            components.html(
            f"""
            <script>
            function setSpeed(speed) {{
                const audio = document.getElementById("ttsAudio");
                if (audio) {{
                    audio.playbackRate = speed;
                    document.getElementById("speedValue").innerText = speed + "x";
                }}
            }}
            </script>
            
            <audio id="ttsAudio" controls style="width:100%; margin-top:10px;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>
            
            <div style="margin-top:12px; color:white;">
                🔈 <b>Playback Speed</b><br>
                <input type="range"
                       min="0.5"
                       max="2.0"
                       step="0.1"
                       value="1"
                       oninput="setSpeed(this.value)">
                <span id="speedValue">1.0x</span>
            </div>
            """,
            height=180
            )
            # ============== Visualisasi =================
            st.markdown("#### 📈 Waveform Visualization")
            
            # Load audio from bytes
            y, sr = librosa.load(
                io.BytesIO(st.session_state["audio_bytes"]),
                sr=None
            )
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(y)
            ax.set_title("Waveform (Amplitude vs Time)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            
            st.pyplot(fig)

            st.markdown("#### 🎼 Mel-Spectrogram")

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=80,
                fmax=8000
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            img = ax.imshow(
                mel_db,
                aspect="auto",
                origin="lower",
                interpolation="nearest"
            )
            ax.set_title("Mel-Spectrogram (dB)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel Frequency")
            
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)

            st.markdown("#### 🎚️ Pitch Contour (Fundamental Frequency)")

            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7")
            )
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(f0, label="F0 (Pitch)")
            ax.set_title("Pitch Contour Over Time")
            ax.set_xlabel("Frames")
            ax.set_ylabel("Frequency (Hz)")
            ax.legend()
            
            st.pyplot(fig)

            audio_duration = librosa.get_duration(y=y, sr=sr)
            rtf = round(st.session_state["latency"] / audio_duration, 3)
            
            st.metric(
                label="⚡ Real-Time Factor (RTF)",
                value=rtf
            )
            
            st.markdown("#### 📊 Latency vs Input Length")
            
            fig, ax = plt.subplots()
            ax.scatter(
                [char_count],
                [st.session_state["latency"]],
                s=80
            )
            ax.set_xlabel("Number of Characters")
            ax.set_ylabel("Inference Latency (s)")
            ax.set_title("Latency Scaling")
            
            st.pyplot(fig)

            # ================= METRICS =================
            st.markdown("#### 📊 Inference Metrics")

            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.metric(
                    label="⏱️ Inference Latency",
                    value=f"{st.session_state['latency']} s"
                )

            with col_m2:
                estimated_duration = round(char_count * 0.045, 2)
                st.metric(
                    label="🎵 Estimated Audio Duration",
                    value=f"~ {estimated_duration} s"
                )

            # ================= DOWNLOAD =================
            st.markdown("#### ⬇️ Download")
            st.download_button(
                label="Download WAV",
                data=st.session_state["audio_bytes"],
                file_name="tts_output.wav",
                mime="audio/wav",
                use_container_width=True
            )

        else:
            st.info("🎧 Audio output akan muncul setelah proses generate.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# ================= ABOUT =================
with tab_about:
    st.markdown('<div class="section-title">🧠 Technology Behind</div>', unsafe_allow_html=True)

    # ---- MODEL CARD ----
    st.markdown("""
    <div class="glass-card">
        <h3>🧠 VITS Architecture</h3>
        <p>
        Sistem ini menggunakan <b>VITS (Variational Inference with Adversarial Learning
        for End-to-End Text-to-Speech)</b>, yaitu model deep learning yang mampu
        menghasilkan suara manusia secara <b>end-to-end</b> tanpa pipeline TTS terpisah.
        </p>
        <ul>
            <li>Text Encoder & Vocoder terintegrasi</li>
            <li>Adversarial training untuk kualitas audio</li>
            <li>Natural dan smooth speech synthesis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ---- DATASET & PREPROCESSING ----
    col1, col2 = st.columns(2)

    # DATASET CARD
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>📦 Dataset</h3>
            <p>
            Model TTS ini dilatih menggunakan <b>Jenny TTS Dataset</b>, dataset
            open-source yang berisi pasangan <b>teks dan audio suara manusia</b>.
            </p>
            <ul>
                <li>Bahasa: Inggris</li>
                <li>Audio berkualitas tinggi</li>
                <li>Cocok untuk pelatihan model TTS</li>
            </ul>
            <p>
            🔗 Dataset Source:<br>
            <a href="https://huggingface.co/datasets/reach-vb/jenny_tts_dataset" target="_blank" style="color:#93c5fd;">
                Hugging Face - Jenny TTS Dataset
            </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # PREPROCESSING CARD
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>⚙️ Data Preprocessing</h3>
            <p>
            Sebelum digunakan untuk training, dataset melalui beberapa tahap
            preprocessing untuk memastikan konsistensi dan kualitas data.
            </p>
            <ul>
                <li>Standarisasi format audio</li>
                <li>Konversi metadata</li>
                <li>Pembersihan teks (text cleaning)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---- PREPROCESSING STEPS ----
    st.markdown("""
    <div class="glass-card">
        <h3>🧪 Preprocessing Pipeline</h3>
        <ol>
            <li>
                <b>Standarisasi Audio</b><br>
                Audio dikonversi ke format TTS-friendly
                (22050 Hz, mono, PCM 16-bit) untuk menjaga konsistensi kualitas suara.
            </li><br>
            <li>
                <b>Metadata Conversion</b><br>
                Metadata diubah dari format TSV ke CSV agar lebih mudah diproses
                oleh sistem training TTS.
            </li><br>
            <li>
                <b>Text Cleaning</b><br>
                Teks dibersihkan dengan menghapus karakter khusus, tanda kutip,
                dan mengubah seluruh teks menjadi lowercase untuk mengurangi noise.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.success("📌 Dataset dan preprocessing yang baik sangat berpengaruh terhadap kualitas suara yang dihasilkan oleh model Text-to-Speech.")

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

    st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    NLP Project • Text to Speech • Streamlit × Hugging Face
</div>
""", unsafe_allow_html=True)





