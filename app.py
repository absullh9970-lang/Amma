# app.py
# --- مساعد الذكاء الاصطناعي "مايند ماستر" (نسخة خفيفة للمنصات المجانية) ---
# --- مصمم للعمل على خططط مثل Render Free Plan ---

import gradio as gr
import torch
import random
import time
import os
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from duckduckgo_search import DDGS
import warnings
warnings.filterwarnings("ignore")

# --- متغيرات عامة ---
chat_llm = None
image_pipe = None
chat_model_loaded = False
image_model_loaded = False

# --- دالة تحميل نموذج الدردشة (تُستدعى مرة واحدة عند أول طلب) ---
def load_chat_model():
    global chat_llm, chat_model_loaded
    if chat_model_loaded:
        return
    try:
        print("🔄 جاري تحميل نموذج الدردشة (قد يستغرق هذا دقيقة واحدة)...")
        # استخدام نموذج أصغر وأخف وزناً لتوفير الذاكرة
        model_id = "TheBloke/TinyLlama-2-1.1B-Chat-v1.0-GGUF"
        model_file = "tinyllama-2-1.1b-chat-v1.0.Q4_K_M.gguf"
        model_path = hf_hub_download(repo_id=model_id, filename=model_file)
        chat_llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=0, verbose=False)
        chat_model_loaded = True
        print("✅ تم تحميل نموذج الدردشة بنجاح!")
    except Exception as e:
        print(f"❌ خطأ في تحميل نموذج الدردشة: {e}")
        chat_model_loaded = False

# --- دالة تحميل نموذج الصور (تُستدعى مرة واحدة عند أول طلب) ---
def load_image_model():
    global image_pipe, image_model_loaded
    if image_model_loaded:
        return
    try:
        print("🎨 جاري تحميل نموذج توليد الصور (قد يستغرق هذا بضع دقائق)...")
        # استخدام نموذج أصغر وأسرع للصور
        image_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
        image_pipe.to("cpu")
        image_model_loaded = True
        print("✅ تم تحميل نموذج الصور بنجاح!")
    except Exception as e:
        print(f"❌ خطأ في تحميل نموذج الصور: {e}")
        image_model_loaded = False

# --- دوال الخدمات ---
def perform_web_search(query):
    try:
        with DDGS() as ddgs:
            results = [r for r, _ in ddgs.text(query, max_results=3)]
        return "\n".join(results)
    except Exception as e:
        return f"فشل البحث: {e}"

def generate_text(message, history, search_enabled, creativity_level):
    if not chat_model_loaded:
        load_chat_model() # حاول التحميل إذا لم يكن محملاً
        if not chat_model_loaded:
            return "نموذج الدردشة غير متوفر حالياً.", history

    # رسالة نظام مبسطة
    prompt = f"
