import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from lorahub.constant import LORA_MODULE_NAMES
# 你的 base model
BASE_MODEL = "google/flan-t5-large"


print("🔵 Downloading base model...")
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🔵 Downloading LoRA modules...")
for lora in LORA_MODULE_NAMES:
    print(f"Downloading {lora}")
    try:
        PeftModel.from_pretrained(model, lora)
    except Exception as e:
        print(f"❌ Failed: {lora} | {e}")

print("✅ All downloads completed.")