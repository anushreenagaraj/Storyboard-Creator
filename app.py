import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
from fpdf import FPDF
import io
import random

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "./local_sd_model",
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    )
    pipe = pipe.to("cpu")
    return pipe

def generate_image(pipe, prompt):
    image = pipe(prompt, height=384, width=384).images[0]
    return image

def create_pdf(images, captions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for img, cap in zip(images, captions):
        img_path = "temp_img.png"
        img.save(img_path)
        pdf.add_page()
        pdf.image(img_path, x=10, y=30, w=180)
        pdf.set_xy(10, 10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, cap or " ")

    # Save to string then encode as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes


def ai_generate_prompts(base_prompt, num):
    variations = [
        f"{base_prompt}, early morning",
        f"{base_prompt}, with animals",
        f"{base_prompt}, sunset view",
        f"{base_prompt}, mysterious atmosphere",
        f"{base_prompt}, celebration scene",
        f"{base_prompt}, peaceful nature",
        f"{base_prompt}, danger ahead",
        f"{base_prompt}, discovering something new",
        f"{base_prompt}, inside a cave",
        f"{base_prompt}, magical transformation"
    ]
    return random.sample(variations, k=num)

# ✅ Frame-by-frame AI prompt assist
def suggest_prompt(frame_number):
    ideas = [
        "A magical discovery in the forest",
        "An unexpected visitor arrives",
        "A dramatic plot twist",
        "Celebration after victory",
        "A calm before the storm",
        "The journey begins",
        "A hidden secret is revealed",
        "The hero faces a challenge",
        "A mysterious place is found",
        "A heartfelt moment between characters"
    ]
    return f"Frame {frame_number + 1}: {random.choice(ideas)}"

pipe = load_model()

st.set_page_config(layout="wide")
st.title("🎬 Storyboard Creator")
st.markdown("Generate a storyboard using your own prompts or AI assistance.")

num_frames = st.number_input("Number of storyboard frames:", min_value=1, max_value=10, step=1)
use_ai = st.checkbox("🤖 Use AI to help generate prompts?")

prompts = []

if use_ai:
    base_idea = st.text_input("Enter your base story idea (AI will generate prompts):", value="")
    if base_idea and st.button("🔮 Generate Prompts with AI"):
        prompts = ai_generate_prompts(base_idea, num_frames)
        for i, p in enumerate(prompts):
            st.text_input(f"AI Prompt for Frame {i + 1}", value=p, key=f"ai_{i}")
else:
    input_mode = st.radio("Choose prompt mode", ["Single Prompt for All Frames", "Separate Prompts Per Frame"])
    if input_mode == "Single Prompt for All Frames":
        main_prompt = st.text_input("Enter prompt for all frames")
        if main_prompt:
            prompts = [main_prompt] * num_frames
    else:
        for i in range(num_frames):
            col1, col2 = st.columns([4, 1])
            with col1:
                prompt = st.text_input(f"Prompt for Frame {i + 1}", key=f"prompt_{i}")
            with col2:
                if st.button(f"✨ AI", key=f"ai_button_{i}"):
                    prompt = suggest_prompt(i)
                    st.session_state[f"prompt_{i}"] = prompt
            prompts.append(st.session_state.get(f"prompt_{i}", ""))

if st.button("🎨 Generate Storyboard Images") and prompts and len(prompts) == num_frames:
    images = []
    captions = []
    with st.spinner("Generating images..."):
        for i, prompt in enumerate(prompts):
            st.subheader(f"🖼 Frame {i + 1}")
            img = generate_image(pipe, prompt)
            st.image(img, caption=f"Prompt: {prompt}", use_container_width=True)
            captions.append(f"Prompt: {prompt}")
            images.append(img)
            captions.append("")

    # Save and download as PDF
    pdf_data = create_pdf(images, captions)
    st.download_button("📄 Download Storyboard as PDF", data=pdf_data, file_name="storyboard.pdf", mime="application/pdf")
