import json, sys, time, torch
from pathlib import Path
root = Path('E:/zalo-challenge')
sys.path.insert(0, str(root / 'public_test'))
from zaic_tools import ToolEnv
from zaic_tools.validate import catalog_tools
from src.agent import LifeMateAgent, LLMBackend
from src.preprocessor_a import ContextLanguagePreprocessorA
from src.router import SemanticToolRouter

cat = json.loads((root / 'public_test/public_test_tools_catalog.json').read_text(encoding='utf-8'))
cat_tools = catalog_tools(cat)
cat_apps = {a['id']: a for a in cat.get('apps', [])}
sample = json.loads(open(root / 'public_test/public_test.jsonl', encoding='utf-8').readline())

print(f"Sample: {sample['id']}")
llm = LLMBackend(
    model_name_or_path="Qwen/Qwen3-4B",
    lora_path=str(root / "models/qwen3-4b-zac-lora-p1-concise"),
    backend="transformers",
    load_in_4bit=True,
)

from src.asr_fallback import SafeContextPreprocessor
_safe = SafeContextPreprocessor()
class Adapt:
    def preprocess(self, t, s): return _safe.preprocess_with_evidence(t, s)[0]

router = SemanticToolRouter(top_k_recommended=4)
agent = LifeMateAgent(
    llm=llm,
    catalog_tools_map=cat_tools,
    max_steps=5,
    preprocessor=Adapt(),
    router=router,
    catalog_apps_map=cat_apps,
)

env = ToolEnv('public_test', sample['id'], root=root)
final_text = "Đặt xe bốn chỗ từ công ty đi khám răng chiều nay, đặt ngay bây giờ nhé."

t0 = time.time()
print("\n>>> Bắt đầu run_sample...")
tc, st = agent.run_sample(env, sample, final_text)
t1 = time.time()

print(f"\n>>> Hoàn tất run_sample trong {t1 - t0:.2f}s!")
print("Tool calls:", tc)
print("State hash:", st)
