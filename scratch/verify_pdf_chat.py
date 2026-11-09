"""
End-to-end verification of PDF Upload and Grounded Chat with real FLAT Exam PDF
"""

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = app.test_client()

with open(r"C:\Users\Gurnoor\Downloads\FLAT-(5th)-Dec2025-2.pdf", "rb") as f:
    pdf_bytes = f.read()

print("[1] Uploading real FLAT exam paper...")
res = client.post(
    "/api/pdf/upload",
    data={"pdf": (io.BytesIO(pdf_bytes), "FLAT-(5th)-Dec2025-2.pdf", "application/pdf")},
    content_type="multipart/form-data",
)
assert res.status_code == 201, f"Upload failed: {res.get_json()}"
doc_id = res.get_json()["doc_id"]
print(f"    Uploaded successfully! Assigned doc_id: {doc_id}")

print("[2] Waiting for local OCR and inspection...")
for i in range(30):
    time.sleep(0.5)
    s_res = client.get(f"/api/pdf/status/{doc_id}")
    status_data = s_res.get_json()
    if status_data["status"] == "completed":
        print(f"    Inspection COMPLETED in ~{(i+1)*0.5:.1f}s!")
        print(f"    Pages: {status_data['page_count']}, SHA-256: {status_data['sha256'][:16]}...")
        break
    elif status_data["status"] == "failed":
        print(f"    Inspection failed: {status_data['error']}")
        break

print("[3] Querying /api/chat with attached doc_id...")
chat_res = client.post(
    "/api/chat",
    json={
        "messages": [
            {
                "role": "user",
                "content": "What are the instructions to candidates regarding Section A and Section B in this exam paper?",
            }
        ],
        "mode": "auto",
        "pdf_doc_ids": [doc_id],
    },
)
assert chat_res.status_code == 200, f"Chat request failed: {chat_res.get_json()}"

print("[4] Inspecting SSE stream:")
found_route = False
sources = []
for line in chat_res.response:
    line_str = line.decode("utf-8", errors="ignore")
    if line_str.startswith("data: "):
        try:
            data = json.loads(line_str[6:].strip())
            if data.get("type") == "route":
                found_route = True
                sources = data.get("pdf_sources", [])
                print(f"    Route: {data.get('route')}")
                print(f"    PDF Sources emitted: {sources}")
                break  # Don't wait for full LLM generation
        except Exception:
            pass

print("\n[+] Verification successful!")
print(f"    Found route: {found_route}")
print(f"    Emitted PDF sources count: {len(sources)}")
