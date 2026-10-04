"""Lightweight local human relevance annotation web server.

Serves a clean, blinded annotation interface on localhost:8050 to review and score
the 496 masked candidate papers, autosaving directly to data/evaluation/judgments/.
"""

import sys
import json
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT

JUDGMENTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "judgments"
BATCH_FILES = [
    "batch1_health.json",
    "batch2_agriculture.json",
    "batch3_economics.json",
    "batch4_climate.json",
]

PORT = 8050

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kenyan Academic Research — Relevance Judgment Workspace</title>
<style>
  :root {
    --bg-primary: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --border: #e2e8f0;
    --brand: #2563eb;
    --brand-hover: #1d4ed8;
    --rel-0: #ef4444;
    --rel-1: #f59e0b;
    --rel-2: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-primary); color: var(--text-primary); line-height: 1.5; padding: 24px; max-width: 1000px; margin: 0 auto; }
  header { margin-bottom: 24px; }
  .header-title { font-size: 20px; font-weight: 700; margin-bottom: 8px; }
  .progress-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  .progress-bar-bg { background: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 8px; margin-bottom: 8px; }
  .progress-bar-fill { background: var(--brand); height: 100%; width: 0%; transition: width 0.3s ease; }
  .progress-stats { display: flex; justify-content: space-between; font-size: 13px; color: var(--text-secondary); }
  .batch-selector { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .batch-btn { padding: 8px 14px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
  .batch-btn.active { background: var(--brand); color: #fff; border-color: var(--brand); }
  
  .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); margin-bottom: 20px; }
  .badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; background: #e0f2fe; color: #0369a1; margin-bottom: 12px; }
  .query-box { background: #f1f5f9; border-left: 4px solid var(--brand); padding: 14px 18px; border-radius: 4px; margin-bottom: 20px; font-size: 16px; font-weight: 600; color: #1e293b; }
  .paper-title { font-size: 18px; font-weight: 700; margin-bottom: 12px; color: #0f172a; }
  .paper-id { font-size: 12px; color: #64748b; font-family: monospace; margin-bottom: 12px; }
  .abstract-box { background: #ffffff; border: 1px solid var(--border); border-radius: 6px; padding: 16px; font-size: 14px; color: var(--text-secondary); line-height: 1.6; max-height: 260px; overflow-y: auto; margin-bottom: 24px; }
  
  .judgment-section { border-top: 1px solid var(--border); padding-top: 20px; }
  .judgment-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #334155; }
  .btn-group { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
  .rel-btn { padding: 14px; border: 2px solid var(--border); background: #ffffff; border-radius: 8px; cursor: pointer; text-align: left; transition: all 0.2s; }
  .rel-btn:hover { border-color: #cbd5e1; background: #f8fafc; }
  .rel-btn.selected-0 { border-color: var(--rel-0); background: #fef2f2; }
  .rel-btn.selected-1 { border-color: var(--rel-1); background: #fffbeb; }
  .rel-btn.selected-2 { border-color: var(--rel-2); background: #f0fdf4; }
  .rel-btn-title { font-weight: 700; font-size: 14px; display: block; margin-bottom: 4px; }
  .rel-btn-desc { font-size: 12px; color: var(--text-secondary); line-height: 1.3; }
  
  .note-input { width: 100%; padding: 10px 14px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; margin-bottom: 20px; }
  .nav-bar { display: flex; justify-content: space-between; align-items: center; }
  .nav-btn { padding: 10px 20px; border: 1px solid var(--border); background: #ffffff; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; }
  .nav-btn:hover { background: #f1f5f9; }
  .nav-btn.primary { background: var(--brand); color: #fff; border-color: var(--brand); }
  .nav-btn.primary:hover { background: var(--brand-hover); }
  .status-tag { font-size: 13px; font-weight: 600; padding: 4px 8px; border-radius: 4px; }
  .status-judged { background: #dcfce7; color: #15803d; }
  .status-pending { background: #fef3c7; color: #b45309; }
</style>
</head>
<body>

<header>
  <div class="header-title">Kenyan Academic Research — Human Relevance Judgment Workspace</div>
  <div style="font-size: 13px; color: var(--text-secondary);">
    Masked evaluation interface (System identity, ranks, and scores are strictly blinded).
  </div>
</header>

<div class="progress-card">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <span style="font-size: 14px; font-weight: 600;">Overall Annotation Progress</span>
    <span id="progress-text" style="font-size: 14px; font-weight: 700; color: var(--brand);">0 / 496 (0.0%)</span>
  </div>
  <div class="progress-bar-bg">
    <div id="progress-bar-fill" class="progress-bar-fill"></div>
  </div>
  <div class="progress-stats">
    <span id="batch-stats">Batch: 0 / 0</span>
    <span id="keyboard-help">Keyboard Shortcuts: <b>0</b>, <b>1</b>, <b>2</b> to select & advance | <b>Left/Right</b> to navigate</span>
  </div>
</div>

<div class="batch-selector" id="batch-buttons"></div>

<div class="card" id="annotation-card">
  <div style="display: flex; justify-content: space-between; align-items: flex-start;">
    <span class="badge" id="item-domain">Domain</span>
    <span id="item-status" class="status-tag status-pending">Pending</span>
  </div>
  
  <div class="query-box" id="item-query">Query Text</div>
  
  <div class="paper-title" id="item-title">Paper Title</div>
  <div class="paper-id" id="item-paper-id">Work ID: W...</div>
  
  <div class="abstract-box" id="item-abstract">Abstract text loading...</div>
  
  <div class="judgment-section">
    <div class="judgment-title">Assign Relevance Judgment:</div>
    <div class="btn-group">
      <button class="rel-btn" id="btn-rel-0" onclick="setRelevance(0)">
        <span class="rel-btn-title" style="color: var(--rel-0);">[0] Irrelevant</span>
        <span class="rel-btn-desc">Does not address information need; off-topic.</span>
      </button>
      <button class="rel-btn" id="btn-rel-1" onclick="setRelevance(1)">
        <span class="rel-btn-title" style="color: var(--rel-1);">[1] Partially Relevant</span>
        <span class="rel-btn-desc">Related domain/topic but does not directly address need.</span>
      </button>
      <button class="rel-btn" id="btn-rel-2" onclick="setRelevance(2)">
        <span class="rel-btn-title" style="color: var(--rel-2);">[2] Highly Relevant</span>
        <span class="rel-btn-desc">Directly, substantively addresses query in Kenya.</span>
      </button>
    </div>
    
    <input type="text" id="item-notes" class="note-input" placeholder="Optional assessor note (press Enter to save note)..." onchange="saveCurrentItem()">
  </div>
  
  <div class="nav-bar">
    <button class="nav-btn" onclick="prevItem()">&larr; Previous</button>
    <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary);" id="item-counter">Item 1 of 100</span>
    <div style="display: flex; gap: 8px;">
      <button class="nav-btn" onclick="jumpNextUnjudged()">Next Unjudged &rarr;</button>
      <button class="nav-btn primary" onclick="nextItem()">Next &rarr;</button>
    </div>
  </div>
</div>

<script>
let currentBatchId = "batch1_health";
let batches = {};
let currentItems = [];
let currentIndex = 0;

async function loadAllData() {
  const resp = await fetch('/api/data');
  batches = await resp.json();
  renderBatchButtons();
  selectBatch(currentBatchId);
}

function renderBatchButtons() {
  const container = document.getElementById('batch-buttons');
  container.innerHTML = '';
  
  for (const [bId, bData] of Object.entries(batches)) {
    const btn = document.createElement('button');
    btn.className = `batch-btn ${bId === currentBatchId ? 'active' : ''}`;
    const completed = bData.items.filter(i => i.relevance !== null).length;
    btn.innerText = `${bData.domain} (${completed}/${bData.items.length})`;
    btn.onclick = () => selectBatch(bId);
    container.appendChild(btn);
  }
  updateGlobalProgress();
}

function selectBatch(bId) {
  currentBatchId = bId;
  currentItems = batches[bId].items;
  currentIndex = 0;
  
  // Find first unjudged
  const firstUnjudged = currentItems.findIndex(i => i.relevance === null);
  if (firstUnjudged !== -1) currentIndex = firstUnjudged;
  
  renderBatchButtons();
  renderCurrentItem();
}

function updateGlobalProgress() {
  let total = 0;
  let completed = 0;
  for (const bData of Object.values(batches)) {
    total += bData.items.length;
    completed += bData.items.filter(i => i.relevance !== null).length;
  }
  const pct = total > 0 ? (completed / total * 100).toFixed(1) : 0;
  document.getElementById('progress-text').innerText = `${completed} / ${total} (${pct}%)`;
  document.getElementById('progress-bar-fill').style.width = `${pct}%`;
}

function renderCurrentItem() {
  if (!currentItems || currentItems.length === 0) return;
  const item = currentItems[currentIndex];
  
  document.getElementById('item-domain').innerText = `${batches[currentBatchId].domain} (${item.query_id})`;
  document.getElementById('item-query').innerText = item.query_text;
  document.getElementById('item-title').innerText = item.title;
  document.getElementById('item-paper-id').innerText = `Candidate Work ID: ${item.paper_id}`;
  document.getElementById('item-abstract').innerText = item.abstract || "(No abstract available for this publication in OpenAlex).";
  document.getElementById('item-notes').value = item.notes || "";
  
  document.getElementById('item-counter').innerText = `Item ${currentIndex + 1} of ${currentItems.length}`;
  
  const statusElem = document.getElementById('item-status');
  if (item.relevance !== null) {
    statusElem.className = 'status-tag status-judged';
    statusElem.innerText = `Judged: [${item.relevance}]`;
  } else {
    statusElem.className = 'status-tag status-pending';
    statusElem.innerText = 'Pending';
  }
  
  // Update buttons
  document.getElementById('btn-rel-0').className = `rel-btn ${item.relevance === 0 ? 'selected-0' : ''}`;
  document.getElementById('btn-rel-1').className = `rel-btn ${item.relevance === 1 ? 'selected-1' : ''}`;
  document.getElementById('btn-rel-2').className = `rel-btn ${item.relevance === 2 ? 'selected-2' : ''}`;
}

async function setRelevance(rel) {
  const item = currentItems[currentIndex];
  item.relevance = rel;
  item.notes = document.getElementById('item-notes').value;
  
  await saveCurrentItem();
  renderCurrentItem();
  renderBatchButtons();
  
  // Auto-advance
  if (currentIndex < currentItems.length - 1) {
    currentIndex++;
    renderCurrentItem();
  }
}

async function saveCurrentItem() {
  const item = currentItems[currentIndex];
  item.notes = document.getElementById('item-notes').value;
  
  await fetch('/api/save_item', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      batch_id: currentBatchId,
      query_id: item.query_id,
      paper_id: item.paper_id,
      relevance: item.relevance,
      notes: item.notes
    })
  });
}

function prevItem() {
  if (currentIndex > 0) {
    currentIndex--;
    renderCurrentItem();
  }
}

function nextItem() {
  if (currentIndex < currentItems.length - 1) {
    currentIndex++;
    renderCurrentItem();
  }
}

function jumpNextUnjudged() {
  const nextIdx = currentItems.findIndex((item, idx) => idx > currentIndex && item.relevance === null);
  if (nextIdx !== -1) {
    currentIndex = nextIdx;
    renderCurrentItem();
  } else {
    const fromStart = currentItems.findIndex(item => item.relevance === null);
    if (fromStart !== -1) {
      currentIndex = fromStart;
      renderCurrentItem();
    } else {
      alert("All items in this batch are completed!");
    }
  }
}

document.addEventListener('keydown', (e) => {
  if (document.activeElement === document.getElementById('item-notes')) return;
  if (e.key === '0') setRelevance(0);
  if (e.key === '1') setRelevance(1);
  if (e.key === '2') setRelevance(2);
  if (e.key === 'ArrowLeft') prevItem();
  if (e.key === 'ArrowRight') nextItem();
});

loadAllData();
</script>
</body>
</html>
"""


class AnnotationServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/data":
            batches_data = {}
            for b_file in BATCH_FILES:
                b_path = JUDGMENTS_DIR / b_file
                if b_path.exists():
                    with open(b_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        batches_data[data["batch_id"]] = data

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(batches_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save_item":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            b_id = payload["batch_id"]
            q_id = payload["query_id"]
            p_id = payload["paper_id"]
            rel = payload.get("relevance")
            notes = payload.get("notes", "")

            # Match batch filename
            matched_file = None
            for b_file in BATCH_FILES:
                if b_file.startswith(b_id):
                    matched_file = JUDGMENTS_DIR / b_file
                    break

            if matched_file and matched_file.exists():
                with open(matched_file, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)

                updated = False
                for item in batch_data.get("items", []):
                    if item["query_id"] == q_id and item["paper_id"] == p_id:
                        item["relevance"] = int(rel) if rel is not None else None
                        item["notes"] = notes
                        updated = True
                        break

                if updated:
                    batch_data["completed_items"] = sum(
                        1 for i in batch_data["items"] if i.get("relevance") is not None
                    )
                    with open(matched_file, "w", encoding="utf-8") as f:
                        json.dump(batch_data, f, indent=2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server():
    server = HTTPServer(("127.0.0.1", PORT), AnnotationServerHandler)
    print("=" * 70)
    print(f"Human Relevance Annotation Workspace Live at: http://localhost:{PORT}")
    print("Press Ctrl+C in terminal to stop server.")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
