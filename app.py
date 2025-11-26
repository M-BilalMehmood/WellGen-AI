#!/usr/bin/env python3
"""
WellGen AI — Modern Minimalist Chatbot UI
Single file Streamlit app with ChatGPT-style interface
- Profile setup integrated into chat flow
- Clean, minimal chat interface
- RAG text generation + LoRA image generation
"""

import streamlit as st
from pathlib import Path
import time
from dotenv import load_dotenv
import argparse
from PIL import Image
import uuid
import re

load_dotenv()

import sys
sys.path.append('/mnt/c/Users/bilal/OneDrive/Documents/University/Gen AI/Project/wellgen-ai')

from src.text_gen.wellgen_rag import WellGenRAG
from src.image_gen.inference import generate_images

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="WellGen AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# LOAD AI
# ---------------------------------------------------------
@st.cache_resource
def load_text_ai():
    return WellGenRAG(use_rag=True)

text_ai = load_text_ai()

# ---------------------------------------------------------
# MODERN MINIMALIST CSS (DARK THEME)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main container - Dark Theme */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
        padding: 0 !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem 100px 1rem; /* Added bottom padding for input area */
    }
    
    /* Message bubble styling */
    .message-wrapper {
        display: flex;
        margin-bottom: 1rem;
        animation: slideIn 0.3s ease-out;
    }
    
    .message-bubble {
        max-width: 75%;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        line-height: 1.6;
        word-wrap: break-word;
        font-size: 1rem;
    }
    
    .message-bubble.assistant {
        background: #262730;
        color: #e0e0e0;
        border: 1px solid #363940;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .message-bubble.user {
        background: #2e7d32; /* Motivating Green */
        color: white;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Right-align user messages */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .user-avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background-color: #1f2937;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 10px;
        font-size: 1.2rem;
        border: 1px solid #374151;
        flex-shrink: 0;
    }
    
    /* Input area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 1.5rem 0;
        background: #0e1117;
        border-top: 1px solid #363940;
        z-index: 100;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        border-bottom: 1px solid #363940;
        background: #0e1117;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        margin: 0.5rem 0 0 0;
        color: #9ca3af;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Profile badge */
    .profile-badge {
        background: #1f2937;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #d1d5db;
        margin-top: 1rem;
        text-align: center;
        border: 1px solid #374151;
    }
    
    .profile-badge strong {
        color: #4ade80; /* Green accent */
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4b5563;
    }
    
    /* Images */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .image-item {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #374151;
        background: #1f2937;
    }
    
    /* Setup form styling */
    .setup-form {
        background: #1f2937;
        padding: 2rem;
        border-radius: 12px;
        max-width: 600px;
        margin: 2rem auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid #374151;
    }
    
    .setup-form h2 {
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    
    .setup-form label {
        color: #e0e0e0;
        font-weight: 500;
        margin-top: 1rem;
        display: block;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #363940;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #d1d5db;
    }
    
    /* Collage Style Grid */
    .collage-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    .collage-grid img {
        width: 100%;
        border-radius: 10px;
        object-fit: cover;
        aspect-ratio: 1/1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 'setup'

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "diet_plan" not in st.session_state:
    st.session_state.diet_plan = None

if "body_images" not in st.session_state:
    st.session_state.body_images = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile_complete" not in st.session_state:
    st.session_state.profile_complete = False

if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def format_diet_plan(plan_text):
    """Format diet plan for better readability."""
    formatted = plan_text.replace('*', '**')
    
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        formatted = formatted.replace(f'**{day}**', f'\n### {day}')
    
    formatted = formatted.replace('**Breakfast:**', '\n**Breakfast:**')
    formatted = formatted.replace('**Lunch:**', '\n**Lunch:**')
    formatted = formatted.replace('**Dinner:**', '\n**Dinner:**')
    formatted = formatted.replace('**Snacks:**', '\n**Snacks:**')
    
    formatted = formatted.replace('**Expected Results in 8 Weeks:**', '\n\n## Expected Results')
    formatted = formatted.replace('**Body Parts Most Affected by This Diet:**', '\n\n## Targeted Areas')
    formatted = formatted.replace('**5 Key Success Tips Based on Nutrition Science:**', '\n\n## Success Tips')
    formatted = formatted.replace('**Important Warnings:**', '\n\n## Important Notes')
    formatted = formatted.replace('**Next Steps:**', '\n\n## Next Steps')
    
    formatted = formatted.replace('•', '\n-')
    formatted = '\n'.join(line.strip() for line in formatted.split('\n') if line.strip())
    
    return formatted

def extract_body_parts(diet_plan):
    """Extract affected body parts from diet plan by parsing various possible formats."""
    import re
    
    # Try multiple patterns to find body parts section
    patterns = [
        r"## Body Parts Most Affected by This Diet:\s*\n(.*?)(?=\n##|\n\n##|\Z)",
        r"Body parts most affected by this diet[:\-]?\s*\n?(.*?)(?=\n\n|\n[A-Z]|\Z)",
        r"Body parts affected[:\-]?\s*\n?(.*?)(?=\n\n|\n[A-Z]|\Z)",
        r"Affected body parts[:\-]?\s*\n?(.*?)(?=\n\n|\n[A-Z]|\Z)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, diet_plan, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            # Extract items after - or • or numbers
            parts = re.findall(r"[-•]\s*([^-\n•]+)", section)
            if not parts:
                # Try to split by commas or newlines
                parts = [part.strip() for part in re.split(r'[,\n]', section) if part.strip()]
            
            parts = [part.strip().lower() for part in parts if part.strip()]
            
            # Filter to only allowed body parts
            allowed_parts = ['biceps', 'triceps', 'back', 'legs', 'shoulders', 'chest', 'belly']
            filtered_parts = [part for part in parts if any(allowed in part for allowed in allowed_parts)]
            
            if filtered_parts:
                return filtered_parts
    
    # If no section found, try to find any mention of body parts in the plan
    all_parts = re.findall(r'\b(biceps|triceps|back|legs|shoulders|chest|belly)\b', diet_plan, re.IGNORECASE)
    if all_parts:
        return list(set(part.lower() for part in all_parts))
    
    # Default fallback
    return ['chest', 'back', 'legs']  # Common body parts for general fitness

def generate_ai_image(prompt, output_dir="generated_streamlit"):
    """Generate image using LoRA."""
    args = argparse.Namespace(
        base_model="SG161222/Realistic_Vision_V5.1_noVAE",
        lora_path="model/full_lora_finetune",
        prompt=prompt,
        negative_prompt="low quality, deformed, bad anatomy",
        output_dir=output_dir,
        num_images=1,
        steps=30,
        guidance=7.5,
        seed=None
    )
    
    generate_images(args)
    
    original_path = Path(output_dir) / "generated_0000.png"
    if original_path.exists():
        unique_filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        unique_path = Path(output_dir) / unique_filename
        original_path.rename(unique_path)
        return str(unique_path)
    return None

def stream_response(text):
    """Stream text word by word."""
    words = text.split()
    output = ""
    for w in words:
        output += w + " "
        yield output
        time.sleep(0.02)

# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------
def render_header():
    """Render minimalist header."""
    st.markdown("""
    <div class="header">
        <h1>✨ WellGen AI</h1>
        <p>Your AI Wellness Coach</p>
    </div>
    """, unsafe_allow_html=True)

def render_profile_badge(profile):
    """Render user profile badge."""
    if profile:
        st.markdown(f"""
        <div class="profile-badge">
            <strong>{profile['age']}y</strong> • <strong>{profile['weight']}kg</strong> • 
            <strong>{profile['goal'].replace('_', ' ').title()}</strong> • 
            <strong>{profile['cuisine'].title()}</strong> cuisine
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# SETUP FLOW (INTEGRATED INTO CHAT)
# ---------------------------------------------------------
def show_initial_setup():
    """Show profile setup integrated into chat."""
    render_header()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 👋 Let's Get Started")
        st.markdown("Tell me about yourself and I'll create a personalized wellness plan.")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        with col_left:
            height = st.number_input("Height (cm)", 120, 220, 175, label_visibility="visible")
            age = st.number_input("Age", 18, 100, 25, label_visibility="visible")
            allergies = st.text_input("Allergies", "none", label_visibility="visible")
        
        with col_right:
            weight = st.number_input("Weight (kg)", 40, 200, 70, label_visibility="visible")
            gender = st.selectbox("Gender", ["Male", "Female"], label_visibility="visible")
            cuisine = st.selectbox("Cuisine", ["Any", "Asian", "Mediterranean", "Indian", "Pakistani", "Italian"], label_visibility="visible")
        
        goal = st.selectbox("Your Goal", ["Weight Loss", "Muscle Gain", "Maintenance"], label_visibility="visible")
        
        st.divider()
        
        if st.button("🚀 Create My Plan", use_container_width=True, type="primary"):
            st.session_state.user_profile = {
                "height": height,
                "age": age,
                "weight": weight,
                "gender": gender.lower(),
                "goal": goal.lower().replace(" ", "_"),
                "allergies": allergies,
                "cuisine": cuisine.lower()
            }
            st.session_state.step = 'generating'
            st.session_state.profile_complete = True
            st.rerun()

def show_chat_interface():
    """Show the main chat interface."""
    render_header()
    render_profile_badge(st.session_state.user_profile)
    
    # -----------------------------------------------------
    # SIDEBAR: Diet Plan & Visualizations
    # -----------------------------------------------------
    with st.sidebar:
        st.markdown("### 📋 My Wellness Plan")
        
        # Diet Plan Section
        with st.expander("🍽️ Diet Plan", expanded=True):
            if st.session_state.diet_plan:
                st.markdown(st.session_state.diet_plan)
        
        st.divider()
        
        # Body Visualizations Section
        st.markdown("### 🎨 Body Visualizations")
        
        if st.session_state.body_images:
            # Display generated images in sidebar
            for part, img_path in st.session_state.body_images:
                st.image(img_path, caption=part, use_container_width=True)
        else:
            st.info("No visualizations generated yet.")

    # -----------------------------------------------------
    # MAIN CHAT AREA
    # -----------------------------------------------------
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Messages container
    messages_container = st.container()
    
    with messages_container:
        if not st.session_state.messages:
            # Welcome message
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #9ca3af;">
                <h3 style="margin-top: 3rem; color: #ffffff;">👋 Welcome to WellGen AI</h3>
                <p>I'm ready to help you with your wellness journey. Ask me anything about your diet, fitness goals, or health!</p>
                <p style="font-size: 0.8rem; margin-top: 1rem; color: #6b7280;">Try typing <code>/imagine a futuristic gym</code> to generate images!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display messages
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message-container">
                        <div class="message-bubble user">{msg["content"]}</div>
                        <div class="user-avatar">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.chat_message(msg["role"]):
                        st.markdown(f'<div class="message-bubble assistant">{msg["content"]}</div>', unsafe_allow_html=True)
                        
                    if 'images' in msg:
                        # Display images in collage grid
                        cols = st.columns(2)
                        for i, img_path in enumerate(msg['images']):
                            with cols[i % 2]:
                                st.image(img_path, use_container_width=True)
    
    # Anchor for auto-scrolling
    st.markdown('<div id="end-of-chat"></div>', unsafe_allow_html=True)
    
    # Auto-scroll script
    st.markdown("""
    <script>
        var element = window.parent.document.getElementById('end-of-chat');
        if (element) {
            element.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});
        }
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    user_input = st.chat_input("Ask about your plan or type /imagine...", key="chat_input")
            
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately with custom HTML
        st.markdown(f"""
        <div class="user-message-container">
            <div class="message-bubble user">{user_input}</div>
            <div class="user-avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for /imagine command
        if user_input.strip().startswith("/imagine"):
            prompt = user_input.replace("/imagine", "").strip()
            if not prompt:
                response = "Please provide a prompt after /imagine. Example: `/imagine a healthy salad bowl`"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            with st.chat_message("assistant"):
                with st.spinner(f"🎨 Imagining '{prompt}'..."):
                    # Generate 2 images for collage effect
                    generated_images = []
                    for i in range(2):
                        img_path = generate_ai_image(prompt, "generated_chat")
                        if img_path:
                            generated_images.append(img_path)
                    
                    response = f"Here is what I imagined for: **{prompt}**"
                    st.markdown(f'<div class="message-bubble assistant">{response}</div>', unsafe_allow_html=True)
                    
                    # Display in collage
                    cols = st.columns(2)
                    for i, img_path in enumerate(generated_images):
                        with cols[i % 2]:
                            st.image(img_path, use_container_width=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "images": generated_images
                    })
        
        # Check for /visualhelp command
        elif user_input.strip() == "/visualhelp":
            with st.chat_message("assistant"):
                with st.spinner("🎨 Generating visual assistance..."):
                    last_user_msg = ""
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user" and msg["content"] != "/visualhelp":
                            last_user_msg = msg["content"]
                            break
                    
                    prompt = f"Provide visual help for this question: '{last_user_msg}'. Include image prompts in square brackets like [anatomical illustration of biceps]."
                    response = text_ai.chat(prompt)
                    
                    image_prompts = re.findall(r'\[([^\]]+)\]', response)
                    if not image_prompts:
                        image_prompts = [response[:100] + " anatomical illustration"]
                    
                    generated_images = []
                    for img_prompt in image_prompts[:2]:
                        img_path = generate_ai_image(img_prompt, "generated_help")
                        if img_path:
                            generated_images.append(img_path)
                    
                    st.markdown(f'<div class="message-bubble assistant">{response}</div>', unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    for i, img_path in enumerate(generated_images):
                        with cols[i % 2]:
                            st.image(img_path, use_container_width=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "images": generated_images
                    })
        else:
            # Normal chat response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                raw_response = text_ai.chat(user_input)
                streamed = ""
                for chunk in stream_response(raw_response):
                    streamed = chunk
                    placeholder.markdown(f'<div class="message-bubble assistant">{streamed}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": raw_response})
        
        st.rerun()

def show_generating():
    """Show generation progress and perform the actual generation."""
    render_header()
    render_profile_badge(st.session_state.user_profile)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### ⏳ Creating Your Wellness Plan")
        
        # Progress bar
        progress = st.progress(0)
        
        # Step 1: Generate diet plan
        progress.progress(25)
        st.markdown("📋 Analyzing your profile...")
        
        if not st.session_state.diet_plan:
            with st.spinner("🍽️ Generating diet plan..."):
                plan = text_ai.generate_diet_plan(st.session_state.user_profile)
                st.session_state.diet_plan = format_diet_plan(plan)
        
        progress.progress(50)
        st.markdown("🍽️ Diet plan created!")
        
        # Step 2: Generate body images (RESTORED AUTO-GENERATION)
        if not st.session_state.body_images:
            with st.spinner("🎨 Creating body visualizations..."):
                affected_parts = extract_body_parts(st.session_state.diet_plan)
                images = []
                
                # Generate body part visualizations for diet app
                for part in affected_parts:
                    gender = st.session_state.user_profile.get('gender', 'male')
                    
                    # Anatomical body part visualization for diet app
                    prompt = f"anatomical illustration of {gender} {part}, fitness body diagram, clean medical style, highlighted muscle area"
                    
                    print(f"🖼️  Generating: {part}")
                    
                    image_path = generate_ai_image(prompt, "generated_body_parts")
                    if image_path:
                        images.append((part.title(), image_path))
                
                st.session_state.body_images = images
        
        progress.progress(100)
        st.markdown("✅ All done!")
        
        # Toast notification for diet plan
        st.toast("📋 Diet Plan Ready! Open the sidebar (top-left arrow) to view it.", icon="🎉")
        
        # Transition to chat
        time.sleep(2)
        st.session_state.step = 'chat'
        st.rerun()

def stream_response(text):
    """Stream text word by word."""
    words = text.split()
    output = ""
    for w in words:
        output += w + " "
        yield output
        time.sleep(0.02)

# ---------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------
if st.session_state.step == 'setup':
    show_initial_setup()
elif st.session_state.step == 'generating':
    show_generating()
elif st.session_state.step == 'chat':
    show_chat_interface()