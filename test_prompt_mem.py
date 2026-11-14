import json, sys, time, torch
from pathlib import Path
root = Path('E:/zalo-challenge')
sys.path.insert(0, str(root / 'public_test'))
from zaic_tools.validate import catalog_tools
from src.prompt_builder import build_initial_messages
from src.router import SemanticToolRouter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("1. Preparing prompt...")
cat = json.loads((root / 'public_test/public_test_tools_catalog.json').read_text(encoding='utf-8'))
cat_tools = catalog_tools(cat)
cat_apps = {a['id']: a for a in cat.get('apps', [])}
sample = json.loads(open(root / 'public_test/public_test.jsonl', encoding='utf-8').readline())
router = SemanticToolRouter(top_k_recommended=4)
msgs = build_initial_messages(
    sample=sample,
    final_user_text='Xem lịch hôm nay',
    catalog_tools_map=cat_tools,
    router=router,
    catalog_apps_map=cat_apps,
)

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
input_ids = tok.encode(prompt, return_tensors='pt')
print(f"Prompt length: {input_ids.shape[1]} tokens")

print("2. Loading 4-bit model with SDPA...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B',
    quantization_config=bnb,
    device_map='auto',
    torch_dtype=torch.float16,
    attn_implementation='sdpa',
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, str(root / 'models/qwen3-4b-zac-lora-p1-concise'))
model.eval()

vram_load = torch.cuda.memory_allocated() / (1024**3)
print(f"VRAM after model load: {vram_load:.2f} GB")

input_ids = input_ids.to(model.device)
print("3. Generating first token...")
t0 = time.time()
with torch.no_grad():
    out = model.generate(
        input_ids,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
t1 = time.time()
vram_gen = torch.cuda.memory_allocated() / (1024**3)
vram_max = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Generated 30 tokens in {t1 - t0:.2f}s (Speed: {30 / (t1 - t0):.1f} tok/s)")
print(f"VRAM during gen: {vram_gen:.2f} GB | Peak VRAM: {vram_max:.2f} GB / 4.0 GB")
reply = tok.decode(out[0][input_ids.shape[1]:])
print("Output:", reply)
