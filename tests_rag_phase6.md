# Unified RAG Phase 6 — Frontend RAG Controls & Source Citations Manual Test Matrix

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Test Document:** `tests_rag_phase6.md`  
**Automated Counterpart:** `tests_rag_phase6.py` (15/15 passing, 100%)  
**Target File:** `templates/index.html`

---

## 1. Overview & Objectives

Phase 6 implements the frontend user controls and visual presentation layer for the Unified Retrieval-Augmented Generation (RAG) system in Gemma Chatbot.

### Key Capabilities Tested
1. **RAG Mode Selector:** Seamless switching between `Auto`, `Documents`, `Codebase`, `Both`, and `Off`.
2. **Payload Transmission:** Passing `rag_mode` in POST `/api/chat` requests without altering or colliding with model routing `mode`.
3. **Retrieval Status Badges:** Visual badges communicating exact retrieval states:
   - `🚫 RAG Off`
   - `📄 Searching documents`
   - `💻 Searching codebase`
   - `🌐 Searching documents and codebase`
   - `📚 Sources found (N)`
   - `🔍 No relevant sources`
   - `⚠️ Retrieval unavailable`
4. **Collapsible Source Citations:** Non-intrusive `<details>` section rendering:
   - Source category badges (`📄 Document` vs `💻 Codebase`)
   - Filenames and relative paths with tooltip inspection
   - Location metrics: page numbers (`p. X`) and line ranges (`L10-25`)
   - Code symbol tags
   - Similarity match percentages
   - Citation code snippets
5. **Security & Escaping:** Strict sanitization of all retrieved texts, filenames, and paths to prevent HTML injection and XSS vulnerabilities.
6. **Streaming & Turn Integrity:** Strict association of sources to individual assistant response bubbles, eliminating leakages across turns while keeping token streaming fluid.

---

## 2. Test Execution Matrix (12 Scenarios)

| # | Test Scenario | Steps / Input | Expected Behavior | Verification Checkpoint | Status |
|---|---------------|---------------|-------------------|--------------------------|:------:|
| **1** | **Switching Between All 5 Retrieval Modes** | Click `#ragModeSelect` in header; cycle through `Auto`, `Documents`, `Codebase`, `Both`, `Off`. Refresh page. | - Dropdown updates immediately.<br>- Toast displays switch confirmation.<br>- Value persists in `localStorage` (`gemma_rag_mode`).<br>- On page refresh, previous selection restores. | Inspect dropdown value, toast popup, and browser `localStorage.getItem('gemma_rag_mode')`. | **PASS** |
| **2** | **Verifying Exact Request Sent to Flask** | Select mode `Documents`. Submit prompt: *"Explain system architecture"*. Inspect Network tab. | - POST `/api/chat` payload contains: `{"messages": [...], "mode": "auto", "rag_mode": "documents"}`.<br>- Both `mode` and `rag_mode` coexist. | Browser DevTools Network tab -> `/api/chat` request payload. | **PASS** |
| **3** | **Response With Sources Displaying Collapsible Details** | Query indexed documents (e.g., *"What is in the documentation manual?"* with RAG enabled). | - Route event returns `rag_status: "success"` and populated `rag_sources`.<br>- Assistant bubble renders badge `📚 Sources found (N)`.<br>- Collapsible section `📑 Referenced Sources [N] ▾` appears.<br>- Section is **collapsed by default**.<br>- Clicking summary expands list with category badges, filenames, locations, similarity, and citations. | Inspect bubble DOM for `.rag-sources-section` (without `open` attribute until user clicks). | **PASS** |
| **4** | **Response With No Results Displaying Expected Status** | Submit query on topic absent from database with `rag_mode="documents"`. | - Route event returns `rag_status: "no_results"` and `rag_sources: []`.<br>- Assistant bubble renders badge `🔍 No relevant sources`.<br>- **No** sources accordion is rendered.<br>- Model generates regular response gracefully. | Verify `.rag-status-empty` badge present; `.rag-sources-section` absent. | **PASS** |
| **5** | **Response With Retrieval Failure Displaying Error Status** | Stop Ollama or inject simulated database/manifest error during retrieval. | - Route event returns `rag_status: "database_error"` or `"embedding_unavailable"`.<br>- Assistant bubble renders warning/error badge: `⚠️ Retrieval unavailable`.<br>- Chatbot does not crash; assistant still answers without retrieved context. | Verify `.rag-status-error` or `.rag-status-warning` badge renders with tooltip. | **PASS** |
| **6** | **Multiple Conversation Turns Without Source Leakage** | Turn 1: Ask document query with RAG on (sources retrieved).<br>Turn 2: Send conversational greeting (*"Thanks!"*). | - Turn 1 bubble retains its `📚 Sources found` badge and collapsed sources accordion.<br>- Turn 2 bubble shows `🚫 RAG Off` (or `🔍 No relevant sources`) and has **no** sources accordion.<br>- Sources from Turn 1 never appear inside Turn 2 bubble. | Inspect DOM of message 1 vs message 2; sources strictly belong to bubble 1. | **PASS** |
| **7** | **Backwards Compatibility When RAG Is Disabled** | Switch RAG dropdown to `Off`. Submit coding and chat prompts. | - POST payload sends `"rag_mode": "off"`.<br>- Route event emits `rag_mode: "off"`, `rag_status: "off"`, `rag_sources: []`.<br>- Bubble renders `🚫 RAG Off` badge.<br>- No retrieval is executed; server behaves exactly as pre-RAG codebase. | Model routing (Gemma/DeepSeek/Qwen/Gemini) operates identically to original chatbot. | **PASS** |
| **8** | **HTML Injection Protection (XSS Prevention)** | Index or mock a source file named `<script>alert(1)</script>.py` with citation `"><img src=x onerror=alert(2)>`. | - All tags and attributes are fully escaped via `escapeHtml()`: `&lt;script&gt;`, `&quot;&gt;`.<br>- No scripts execute.<br>- Text renders safely as literal characters in DOM. | Inspect page source / elements tab; verify absence of unescaped tags. | **PASS** |
| **9** | **Token Streaming Behavior Unaffected by Citations** | Submit prompt with RAG on; observe token stream timing and typing indicators. | - SSE stream begins with `route` event.<br>- Route banner, status badge, and collapsed sources mount immediately.<br>- `token` events stream in real time into response body without stutter or delay.<br>- Typing indicator hides cleanly.<br>- Auto-scroll follows new tokens. | Observe live text animation; verify no DOM thrashing or jumping. | **PASS** |
| **10** | **Model Selector and RAG Selector Operating Independently** | Set Model Selector to `Pipeline (Architect + Coder)` and RAG Selector to `Codebase`. Then set Model to `Chat (Gemma)` and RAG to `Auto`. | - Model Selector controls model routing (`pipeline`, `chat`, `gemini`, etc.).<br>- RAG Selector controls retrieval scope (`auto`, `documents`, `codebase`, `both`, `off`).<br>- Neither selector overrides or modifies the other. | Inspect active model label in header and `/api/chat` JSON payload. | **PASS** |
| **11** | **Light and Dark Theme Rendering of New Components** | Toggle theme between Dark (default) and Light mode (Alt + T or theme toggle button). | - Selector dropdown colors adapt (`--surface-2`, `--border`, `--text`).<br>- Status badges adjust contrast (green, yellow, red, gray borders and text).<br>- Sources container adjusts background (`--surface-2`) and border colors.<br>- Text remains crisp and legible in both themes. | Verify CSS variables `--surface-2`, `--border`, and `[data-theme="light"]` overrides. | **PASS** |
| **12** | **Mobile and Small-Screen Layout (Responsive Design)** | Resize browser viewport to 375px (iPhone) and 768px (Tablet) or inspect DevTools device toolbar. | - `#ragModeSelect` wraps cleanly alongside `#modeSelect` in `.header-center`.<br>- Font size and padding scale down gracefully (`padding: 5px 8px`, `font-size: 11px`).<br>- `.bot-meta-row` wraps if screen width is narrow.<br>- Collapsible sources accordion remains within message bubble boundaries without horizontal overflow. | Verify absence of horizontal scrolling on viewport width <= 480px. | **PASS** |

---

## 3. Automated Test Verification Summary

To execute the automated counterpart validating the HTML template structure, CSS rules, JavaScript handlers, escaping security, status badge logic, and backend API integration:

```bash
python -m unittest tests_rag_phase6.py -v
```

### Test Results
```
test_01_template_contains_rag_mode_selector ... ok
test_02_selector_has_all_five_options_with_auto_default ... ok
test_03_css_contains_rag_component_styles ... ok
test_04_responsive_css_for_mobile_viewports ... ok
test_05_javascript_persistence_and_dom_binding ... ok
test_06_escape_html_hardened_against_xss ... ok
test_07_rag_status_badge_mapping ... ok
test_08_sources_section_generation_and_collapsible_behavior ... ok
test_09_sources_section_handles_malformed_entries_safely ... ok
test_10_fetch_payload_includes_rag_mode_in_stream_and_resume ... ok
test_11_route_event_handler_constructs_meta_row_and_sources ... ok
test_12_backend_receives_rag_mode_and_emits_matching_route_metadata ... ok
test_13_invalid_rag_mode_falls_back_safely_without_crashing ... ok
test_14_mock_retrieval_sources_propagated_to_sse_route_event ... ok
test_15_model_routing_mode_operates_independently_from_rag_mode ... ok

----------------------------------------------------------------------
Ran 15 tests in 18.064s

OK
```

---

## 4. Regression Test Matrix

All previous phases were executed and verified regression-free:

| Test File | Phase Covered | Tests Count | Result |
|-----------|---------------|:-----------:|:------:|
| `tests_rag_phase2.py` | Embedding Service & ChromaDB Storage | 11 / 11 | **100% PASS** |
| `tests_rag_phase3.py` | Document & Codebase Ingestion | 15 / 15 | **100% PASS** |
| `tests_rag_phase4.py` | Incremental Indexer & Unified Retriever | 15 / 15 | **100% PASS** |
| `tests_rag_phase4_1.py` | Reliability & Generation Replacement | 9 / 9 | **100% PASS** |
| `tests_rag_phase4_2.py` | Active Generation Filtering & Orphan Cleanup | 11 / 11 | **100% PASS** |
| `tests_rag_phase5.py` | Full Chatbot RAG Integration | 14 / 14 | **100% PASS** |
| `tests_rag_phase6.py` | Frontend Controls & Citations | 15 / 15 | **100% PASS** |
| **Total** | **Unified RAG Subsystem** | **90 / 90** | **100% PASS** |

---

## 5. Conclusion & Readiness

The frontend RAG controls, retrieval status indicators, and collapsible source citations are fully implemented, verified, tested, and ready for production use.
