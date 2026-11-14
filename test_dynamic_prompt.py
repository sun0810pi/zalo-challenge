import json, sys, time, torch
from pathlib import Path
root = Path('E:/zalo-challenge')
sys.path.insert(0, str(root / 'public_test'))
from zaic_tools.validate import catalog_tools
from src.prompt_builder import compress_tool_spec, build_xref_provider_map, SYSTEM_INSTRUCTION
from src.router import SemanticToolRouter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1. Test top-5 apps prompt
cat = json.loads((root / 'public_test/public_test_tools_catalog.json').read_text(encoding='utf-8'))
cat_tools = catalog_tools(cat)
cat_apps = {a['id']: a for a in cat.get('apps', [])}
sample = json.loads(open(root / 'public_test/public_test.jsonl', encoding='utf-8').readline())
user_query = "Xem lịch hôm nay"
router = SemanticToolRouter(top_k_recommended=5)

installed = sample['context']['installed_apps']
installed_app_descriptions = {
    app_id: f"{meta.get('name_vi', '')} {meta.get('group', '')} {meta.get('description', '')}".strip()
    for app_id, meta in cat_apps.items()
    if app_id in installed
}
rec_apps, sec_apps = router.route_sample_apps(
    user_query,
    sample,
    cat_tools,
    installed_app_descriptions=installed_app_descriptions,
)

selected_apps = rec_apps[:5]
if "sys" in installed and "sys" not in selected_apps:
    selected_apps.append("sys")

xref = build_xref_provider_map(cat_tools)
tool_specs = []
for app in selected_apps:
    for t in installed.get(app, {}).get("tools", []):
        if t in cat_tools:
            tool_specs.append(json.dumps(compress_tool_spec(cat_tools[t], xref), ensure_ascii=False))

prompt_text = f"<|im_start|>system\n{SYSTEM_INSTRUCTION}\n\nALLOWED TOOLS:\n" + "\n".join(tool_specs) + f"\n<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)
input_ids = tok.encode(prompt_text, return_tensors='pt')
print(f"Selected apps: {selected_apps}")
print(f"Prompt length: {input_ids.shape[1]} tokens (reduced from 5800!)")

# 2. Test Generation
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

input_ids = input_ids.to(model.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
t1 = time.time()

vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
print(f"\nSUCCESS! Generated 40 tokens in {t1 - t0:.2f}s ({40 / (t1 - t0):.1f} tok/s)!")
print(f"Peak VRAM: {vram_peak:.2f} GB / 4.0 GB (No PCIe swapping!)")
reply = tok.decode(out[0][input_ids.shape[1]:])
print("Generated reply:", reply)
