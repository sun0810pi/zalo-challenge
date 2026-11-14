import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("=== KIỂM TRA TẢI VÀ NẠP MODEL 4-BIT (BITSANDBYTES) ===")
print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

base_model = "Qwen/Qwen3-4B"
lora_path = r"E:\zalo-challenge\models\qwen3-4b-zac-lora-p1-concise"

print(f"\n1. Đang tải/nạp Base Model '{base_model}' ở chế độ 4-bit NF4...")
t0 = time.time()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print(f"Base model nạp thành công trong {time.time() - t0:.1f}s!")

vram_base = torch.cuda.memory_allocated(0) / (1024**3)
print(f"VRAM sau khi nạp base model: {vram_base:.2f} GB / 4.0 GB")

if Path(lora_path).is_dir():
    print(f"\n2. Nạp LoRA adapter từ: {lora_path}...")
    t1 = time.time()
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"LoRA nạp thành công trong {time.time() - t1:.1f}s!")
    vram_lora = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"VRAM sau khi gắn LoRA: {vram_lora:.2f} GB / 4.0 GB")

model.eval()

print("\n3. Thử nghiệm sinh văn bản (Test generation)...")
messages = [
    {"role": "system", "content": "Bạn là trợ lý LifeMate thông minh."},
    {"role": "user", "content": "Xin chào! Bạn có thể giúp gì cho tôi?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Kết quả sinh thử:")
print(reply)
print("\n=== HOÀN TẤT THÀNH CÔNG! ===")
