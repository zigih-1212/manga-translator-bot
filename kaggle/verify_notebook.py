import json
from pathlib import Path

nb = json.loads(Path("kaggle/setup.ipynb").read_text(encoding="utf-8"))
code = "".join(nb["cells"][1]["source"])
assert "SERVER = " in code
# extract the SERVER literal and eval it back to original server source
marker = "SERVER = "
start = code.index(marker) + len(marker)
end = code.index("\n", start)
literal = code[start:end].strip()
server = eval(literal)
orig = Path("kaggle/server.py").read_text(encoding="utf-8")
if server == orig:
    print("OK: embedded server == kaggle/server.py (%d bytes)" % len(server))
else:
    print("MISMATCH")
    import difflib
    for line in difflib.unified_diff(orig.splitlines(), server.splitlines(), lineterm=""):
        print(line)
# sanity checks: expected endpoint signatures
for needle in [
    "masks: list[UploadFile] | None = File(None)",
    "masks_data: str | None = Form(None)",
    "async def inpaint_batch_endpoint",
    "async def ocr_endpoint",
    "/inpaint_batch",
]:
    assert needle in server, "missing: " + needle
print("OK: endpoint signatures match colab_client format")
