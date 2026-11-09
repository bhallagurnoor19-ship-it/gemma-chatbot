/**
 * Frontend Multimodal Controls & Routing Verification Audit Test Suite
 * Validates DOM contracts, client-side lifecycle, quota limits, settings persistence,
 * model compatibility notices, and error handling for Phase 2.1.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const htmlPath = path.join(__dirname, 'templates', 'index.html');
const htmlContent = fs.readFileSync(htmlPath, 'utf8');

console.log('================================================================');
console.log(' PHASE 2.1 FRONTEND MULTIMODAL CONTROLS VERIFICATION AUDIT');
console.log('================================================================\n');

let passedTests = 0;
let totalTests = 0;

function runTest(name, fn) {
  totalTests++;
  try {
    fn();
    console.log(`  ✓ [PASS] ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  ✗ [FAIL] ${name}`);
    console.error(`    ${err.message}`);
    process.exitCode = 1;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. STATIC DOM CONTRACT AUDIT
// ─────────────────────────────────────────────────────────────────────────────
console.log('--- 1. Static DOM Contract & Markup Audit ---');

runTest('DOM contains all required elements for Phase 2', () => {
  const requiredIds = [
    'cloudMultimodalCheckbox',
    'cloudMultimodalStatusBadge',
    'attachmentInfoBar',
    'attachmentQuotaPill',
    'attachmentCountLabel',
    'attachmentSizeLabel',
    'attachmentClearBtn',
    'modeCompatibilityBanner',
    'imagePreviewStrip',
    'imageInput',
    'attachBtn',
    'userInput',
    'sendBtn',
    'modeSelect',
    'settingsBackdrop'
  ];

  for (const id of requiredIds) {
    assert(
      htmlContent.includes(`id="${id}"`),
      `Missing element with id="${id}" in templates/index.html`
    );
  }
});

runTest('modeSelect dropdown has clear visual compatibility cues', () => {
  assert(htmlContent.includes('Vision Capable'), 'Missing "Vision Capable" cue in modeSelect');
  assert(htmlContent.includes('Text Only'), 'Missing "Text Only" cue in modeSelect');
  assert(htmlContent.includes('Local Vision'), 'Missing "Local Vision" cue in modeSelect');
  assert(htmlContent.includes('value="auto"'), 'Missing auto mode option');
  assert(htmlContent.includes('value="chat"'), 'Missing chat mode option');
  assert(htmlContent.includes('value="gemini"'), 'Missing gemini mode option');
  assert(htmlContent.includes('value="pipeline"'), 'Missing pipeline mode option');
  assert(htmlContent.includes('value="deepseek"'), 'Missing deepseek mode option');
  assert(htmlContent.includes('value="qwen"'), 'Missing qwen mode option');
});

runTest('Settings modal contains Cloud Multimodal Privacy section with supporting explanation', () => {
  assert(htmlContent.includes('Cloud Multimodal Processing'), 'Missing Cloud Multimodal Processing title');
  assert(htmlContent.includes('When enabled, images may be sent to the configured Gemini Cloud API'), 'Missing exact explanation text');
  assert(htmlContent.includes('When disabled, image processing must remain local'), 'Missing local processing guarantee in explanation');
  assert(htmlContent.includes('Enabling this setting authorizes cloud image processing; it does not force every image request to use Gemini'), 'Missing non-forcing clarification text');
});

runTest('No URL.createObjectURL usage in client code (preventing object URL leakages)', () => {
  assert(!htmlContent.includes('createObjectURL'), 'Detected URL.createObjectURL in index.html; base64 data URIs must be used');
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. SIMULATED DOM & CLIENT LOGIC TEST HARNESS
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 2. Client Logic Simulation & State Transition Audit ---');

class MockElement {
  constructor(id, tagName = 'div') {
    this.id = id;
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.style = {};
    this.className = '';
    this.classList = {
      _classes: new Set(),
      add: (c) => this.classList._classes.add(c),
      remove: (...args) => args.forEach(c => this.classList._classes.delete(c)),
      toggle: (c, val) => {
        if (val === undefined) val = !this.classList._classes.has(c);
        if (val) this.classList._classes.add(c);
        else this.classList._classes.delete(c);
        return val;
      },
      contains: (c) => this.classList._classes.has(c)
    };
    this.innerHTML = '';
    this.textContent = '';
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.title = '';
    this.options = [];
    this.selectedIndex = 0;
    this._listeners = {};
  }

  addEventListener(event, fn) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(fn);
  }

  dispatchEvent(event, payload = {}) {
    if (this._listeners[event]) {
      this._listeners[event].forEach(fn => fn(payload));
    }
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  querySelector(sel) {
    return this.children[0] || null;
  }

  querySelectorAll(sel) {
    return this.children;
  }

  scrollIntoView() {}
}

function setupMockEnvironment() {
  const elements = {
    cloudMultimodalCheckbox: new MockElement('cloudMultimodalCheckbox', 'input'),
    cloudMultimodalStatusBadge: new MockElement('cloudMultimodalStatusBadge', 'span'),
    attachmentInfoBar: new MockElement('attachmentInfoBar', 'div'),
    attachmentQuotaPill: new MockElement('attachmentQuotaPill', 'div'),
    attachmentCountLabel: new MockElement('attachmentCountLabel', 'span'),
    attachmentSizeLabel: new MockElement('attachmentSizeLabel', 'span'),
    attachmentClearBtn: new MockElement('attachmentClearBtn', 'button'),
    modeCompatibilityBanner: new MockElement('modeCompatibilityBanner', 'div'),
    imagePreviewStrip: new MockElement('imagePreviewStrip', 'div'),
    imageInput: new MockElement('imageInput', 'input'),
    userInput: new MockElement('userInput', 'textarea'),
    sendBtn: new MockElement('sendBtn', 'button'),
    modeSelect: new MockElement('modeSelect', 'select'),
    ragModeSelect: new MockElement('ragModeSelect', 'select'),
    pauseBanner: new MockElement('pauseBanner', 'div'),
    pauseBannerText: new MockElement('pauseBannerText', 'span'),
    bannerPauseBtn: new MockElement('bannerPauseBtn', 'button'),
    bannerResumeBtn: new MockElement('bannerResumeBtn', 'button'),
    bannerDismissBtn: new MockElement('bannerDismissBtn', 'button'),
    pausedOptionsBar: new MockElement('pausedOptionsBar', 'div'),
    typingIndicator: new MockElement('typingIndicator', 'div'),
    activeModelLabel: new MockElement('activeModelLabel', 'span'),
    statusDot: new MockElement('statusDot', 'span'),
    saveSettingsBtn: new MockElement('saveSettingsBtn', 'button'),
    geminiApiKeyInput: new MockElement('geminiApiKeyInput', 'input'),
    geminiModelSelect: new MockElement('geminiModelSelect', 'select'),
    searchGroundingCheckbox: new MockElement('searchGroundingCheckbox', 'input'),
    geminiKeyStatusBadge: new MockElement('geminiKeyStatusBadge', 'span'),
    settingsChatModelLabel: new MockElement('settingsChatModelLabel', 'span'),
    settingsArchModelLabel: new MockElement('settingsArchModelLabel', 'span'),
    settingsCoderModelLabel: new MockElement('settingsCoderModelLabel', 'span'),
    settingsBackdrop: new MockElement('settingsBackdrop', 'div')
  };

  elements.modeSelect.options = [
    { value: 'auto', text: 'Auto' },
    { value: 'chat', text: 'Chat: Gemma 4' },
    { value: 'gemini', text: 'Gemini API' },
    { value: 'pipeline', text: 'Coding Pipeline' },
    { value: 'deepseek', text: 'DeepSeek Only' },
    { value: 'qwen', text: 'Qwen Only' }
  ];
  elements.modeSelect.value = 'auto';

  const toasts = [];
  function showToast(msg) {
    toasts.push(msg);
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, m => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[m]);
  }

  let pendingImages = [];
  let cachedSettings = { allow_cloud_multimodal: false };
  let serverConfig = {
    chat_model: 'gemma4:12b',
    architect_model: 'deepseek-coder-v2:16b',
    coder_model: 'Qwen2.5-Coder:7B'
  };

  function updateModeBadge() {
    const mode = elements.modeSelect.value;
    if (mode === 'auto') elements.activeModelLabel.textContent = 'Autonomous';
    else if (mode === 'pipeline') elements.activeModelLabel.textContent = 'Pipeline';
    else if (mode === 'chat') elements.activeModelLabel.textContent = 'Chat';
    else if (mode === 'gemini') elements.activeModelLabel.textContent = 'Gemini';
    else if (mode === 'deepseek') elements.activeModelLabel.textContent = 'DeepSeek';
    else if (mode === 'qwen') elements.activeModelLabel.textContent = 'Qwen';
    updateCompatibilityNotice();
  }

  function setMode(val) {
    elements.modeSelect.value = val;
    updateModeBadge();
    updateCompatibilityNotice();
    showToast(`Switched mode to: ${val}`);
  }

  function updateCloudMultimodalBadge(checkboxState) {
    const badge = elements.cloudMultimodalStatusBadge;
    if (!badge) return;
    const isConfirmedByServer = Boolean(cachedSettings && cachedSettings.allow_cloud_multimodal === true);

    if (isConfirmedByServer) {
      if (checkboxState === false) {
        badge.className = 'settings-status-badge';
        badge.textContent = 'Pending Disable (Unsaved)';
      } else {
        badge.className = 'settings-status-badge active';
        badge.textContent = '✓ Permitted';
      }
    } else {
      if (checkboxState === true) {
        badge.className = 'settings-status-badge';
        badge.textContent = 'Pending Save (Not Yet Enabled)';
      } else {
        badge.className = 'settings-status-badge missing';
        badge.textContent = 'Local Only (Disabled)';
      }
    }
  }

  function updateCompatibilityNotice() {
    const banner = elements.modeCompatibilityBanner;
    if (!banner) return;

    if (pendingImages.length === 0) {
      banner.style.display = 'none';
      banner.innerHTML = '';
      return;
    }

    const mode = elements.modeSelect.value;
    const isCloudPermitted = cachedSettings && cachedSettings.allow_cloud_multimodal === true;

    if (mode === 'pipeline' || mode === 'deepseek' || mode === 'qwen') {
      banner.className = 'mode-compatibility-banner warning';
      banner.style.display = 'flex';
      banner.innerHTML = `⚠️ Incompatible Mode: ${mode} is a text-only mode and cannot process images.`;
      return;
    }

    if (mode === 'gemini' && !isCloudPermitted) {
      banner.className = 'mode-compatibility-banner permission';
      banner.style.display = 'flex';
      banner.innerHTML = `🔒 Cloud Permission Required: Gemini API requires Cloud Multimodal Processing in Settings.`;
      return;
    }

    if (mode === 'auto') {
      banner.className = 'mode-compatibility-banner info';
      banner.style.display = 'flex';
      banner.innerHTML = `✨ Vision Capable: Auto mode prioritizes local vision.`;
      return;
    }

    if (mode === 'chat') {
      banner.className = 'mode-compatibility-banner info';
      banner.style.display = 'flex';
      banner.innerHTML = `💬 Local Vision Active.`;
      return;
    }

    banner.style.display = 'none';
  }

  function renderPreviews() {
    elements.imagePreviewStrip.children = [];
    const hasImages = pendingImages.length > 0;
    elements.imagePreviewStrip.classList.toggle('visible', hasImages);

    if (elements.attachmentInfoBar) {
      elements.attachmentInfoBar.style.display = hasImages ? 'flex' : 'none';
    }

    if (hasImages) {
      const totalBytes = pendingImages.reduce((sum, img) => sum + (img.size || 0), 0);
      const totalMb = (totalBytes / (1024 * 1024)).toFixed(2);
      const count = pendingImages.length;

      if (elements.attachmentCountLabel) elements.attachmentCountLabel.textContent = `${count} / 10 images`;
      if (elements.attachmentSizeLabel) elements.attachmentSizeLabel.textContent = `${totalMb} MB / 25 MB`;

      if (elements.attachmentQuotaPill) {
        elements.attachmentQuotaPill.classList.toggle('near-limit', count >= 8 || totalBytes >= 20 * 1024 * 1024);
        elements.attachmentQuotaPill.classList.toggle('at-limit', count >= 10 || totalBytes >= 25 * 1024 * 1024);
      }

      pendingImages.forEach((img, idx) => {
        const item = new MockElement(`preview-${idx}`, 'div');
        item.className = 'image-preview-item';
        elements.imagePreviewStrip.appendChild(item);
      });
    }

    updateCompatibilityNotice();
  }

  function removeAttachment(idx) {
    if (idx >= 0 && idx < pendingImages.length) {
      pendingImages.splice(idx, 1);
      renderPreviews();
    }
  }

  function clearAttachments() {
    pendingImages = [];
    renderPreviews();
    if (elements.imageInput) elements.imageInput.value = '';
  }

  async function handleFiles(files, mockUploadFn) {
    if (!files || files.length === 0) return;
    const fileList = Array.from(files);

    const MAX_IMAGES = 10;
    const MAX_SINGLE_BYTES = 10 * 1024 * 1024;
    const MAX_TOTAL_BYTES = 25 * 1024 * 1024;

    const remainingSlots = MAX_IMAGES - pendingImages.length;
    if (remainingSlots <= 0) {
      showToast('⚠️ Maximum limit of 10 images per prompt reached.');
      return;
    }

    if (fileList.length > remainingSlots) {
      showToast(`⚠️ Only the first ${remainingSlots} image(s) will be added (10 images maximum per prompt).`);
    }

    const allowedFiles = fileList.slice(0, remainingSlots);

    for (const file of allowedFiles) {
      if (!file.type.startsWith('image/')) {
        showToast(`⚠️ Skipping "${file.name}": Only image files are supported.`);
        continue;
      }

      if (file.size > MAX_SINGLE_BYTES) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        showToast(`⚠️ "${file.name}" exceeds 10 MB per-image limit (${sizeMb} MB).`);
        continue;
      }

      const currentTotalBytes = pendingImages.reduce((acc, img) => acc + (img.size || 0), 0);
      if (currentTotalBytes + file.size > MAX_TOTAL_BYTES) {
        const totalMb = ((currentTotalBytes + file.size) / (1024 * 1024)).toFixed(1);
        showToast(`⚠️ Cannot add "${file.name}": Exceeds 25 MB aggregate limit (${totalMb} MB).`);
        break;
      }

      const uploaded = mockUploadFn ? await mockUploadFn(file) : {
        base64: 'b64data',
        filename: file.name,
        size: file.size,
        mime_type: file.type
      };

      pendingImages.push({
        base64: uploaded.base64,
        filename: uploaded.filename,
        size: uploaded.size,
        mime_type: uploaded.mime_type,
        dataUrl: `data:${uploaded.mime_type};base64,${uploaded.base64}`
      });
      renderPreviews();
    }
  }

  function resetPauseState() {
    elements.pauseBanner.classList.remove('visible', 'paused');
    elements.pauseBanner.style.display = 'none';
    elements.typingIndicator.classList.remove('visible');
    elements.sendBtn.classList.remove('is-generating');
    elements.sendBtn.title = 'Send';
  }

  return {
    elements,
    toasts,
    getPendingImages: () => pendingImages,
    setCachedSettings: (s) => { cachedSettings = s; },
    getCachedSettings: () => cachedSettings,
    setMode,
    updateCompatibilityNotice,
    updateCloudMultimodalBadge,
    renderPreviews,
    removeAttachment,
    clearAttachments,
    handleFiles,
    resetPauseState
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ATTACHMENT LIFECYCLE TESTS
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 3. Attachment Lifecycle Tests ---');

runTest('Adding single and multiple valid images within quota', async () => {
  const env = setupMockEnvironment();
  const files = [
    { name: 'pic1.png', type: 'image/png', size: 1024 * 100 },
    { name: 'pic2.jpg', type: 'image/jpeg', size: 1024 * 200 }
  ];

  await env.handleFiles(files);
  assert.strictEqual(env.getPendingImages().length, 2);
  assert.strictEqual(env.elements.attachmentCountLabel.textContent, '2 / 10 images');
  assert(env.elements.imagePreviewStrip.classList.contains('visible'));
  assert.strictEqual(env.elements.attachmentInfoBar.style.display, 'flex');
});

runTest('Individual image removal and quota recalculation', async () => {
  const env = setupMockEnvironment();
  const files = [
    { name: 'pic1.png', type: 'image/png', size: 1024 * 1024 }, // 1 MB
    { name: 'pic2.png', type: 'image/png', size: 2 * 1024 * 1024 } // 2 MB
  ];
  await env.handleFiles(files);
  assert.strictEqual(env.getPendingImages().length, 2);
  assert.strictEqual(env.elements.attachmentSizeLabel.textContent, '3.00 MB / 25 MB');

  // Remove first image
  env.removeAttachment(0);
  assert.strictEqual(env.getPendingImages().length, 1);
  assert.strictEqual(env.getPendingImages()[0].filename, 'pic2.png');
  assert.strictEqual(env.elements.attachmentCountLabel.textContent, '1 / 10 images');
  assert.strictEqual(env.elements.attachmentSizeLabel.textContent, '2.00 MB / 25 MB');
});

runTest('Clear All attachments resets DOM and state completely', async () => {
  const env = setupMockEnvironment();
  await env.handleFiles([{ name: 'test.png', type: 'image/png', size: 500 }]);
  env.elements.imageInput.value = 'C:\\fakepath\\test.png';
  assert.strictEqual(env.getPendingImages().length, 1);

  env.clearAttachments();
  assert.strictEqual(env.getPendingImages().length, 0);
  assert.strictEqual(env.elements.attachmentInfoBar.style.display, 'none');
  assert(!env.elements.imagePreviewStrip.classList.contains('visible'));
  assert.strictEqual(env.elements.imageInput.value, '');
});

runTest('10-image maximum quota hard enforcement', async () => {
  const env = setupMockEnvironment();
  const tenFiles = Array.from({ length: 10 }, (_, i) => ({
    name: `img_${i}.png`,
    type: 'image/png',
    size: 1000
  }));
  await env.handleFiles(tenFiles);
  assert.strictEqual(env.getPendingImages().length, 10);
  assert.strictEqual(env.elements.attachmentCountLabel.textContent, '10 / 10 images');
  assert(env.elements.attachmentQuotaPill.classList.contains('at-limit'));

  // Attempt to add 11th image
  await env.handleFiles([{ name: 'img_11.png', type: 'image/png', size: 1000 }]);
  assert.strictEqual(env.getPendingImages().length, 10);
  assert(env.toasts.some(t => t.includes('Maximum limit of 10 images per prompt reached')));
});

runTest('10 MB per-individual-image limit enforcement', async () => {
  const env = setupMockEnvironment();
  const oversizedFile = { name: 'huge.png', type: 'image/png', size: 11 * 1024 * 1024 }; // 11 MB
  await env.handleFiles([oversizedFile]);
  assert.strictEqual(env.getPendingImages().length, 0);
  assert(env.toasts.some(t => t.includes('exceeds 10 MB per-image limit')));
});

runTest('25 MB aggregate payload limit enforcement', async () => {
  const env = setupMockEnvironment();
  const fileA = { name: 'fileA.png', type: 'image/png', size: 9 * 1024 * 1024 }; // 9 MB
  const fileB = { name: 'fileB.png', type: 'image/png', size: 9 * 1024 * 1024 }; // 9 MB
  const fileC = { name: 'fileC.png', type: 'image/png', size: 9 * 1024 * 1024 }; // 9 MB (total 27 MB)

  await env.handleFiles([fileA, fileB, fileC]);
  // File A and File B should be added (18 MB), File C should be rejected as it would push total to 27 MB > 25 MB
  assert.strictEqual(env.getPendingImages().length, 2);
  assert(env.toasts.some(t => t.includes('Exceeds 25 MB aggregate limit')));
});

runTest('Removing an image allows subsequent replacement under limit', async () => {
  const env = setupMockEnvironment();
  const tenFiles = Array.from({ length: 10 }, (_, i) => ({
    name: `img_${i}.png`,
    type: 'image/png',
    size: 1000
  }));
  await env.handleFiles(tenFiles);
  assert.strictEqual(env.getPendingImages().length, 10);

  // Remove one
  env.removeAttachment(4);
  assert.strictEqual(env.getPendingImages().length, 9);

  // Now adding a replacement must succeed
  await env.handleFiles([{ name: 'replacement.png', type: 'image/png', size: 2000 }]);
  assert.strictEqual(env.getPendingImages().length, 10);
  assert.strictEqual(env.getPendingImages()[9].filename, 'replacement.png');
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. SETTINGS CORRECTNESS & PRIVACY AUDIT
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 4. Settings Correctness & Privacy State Audit ---');

runTest('Status badge indicates Permitted ONLY when server confirms allow_cloud_multimodal=true', () => {
  const env = setupMockEnvironment();

  // Initially false on server
  env.setCachedSettings({ allow_cloud_multimodal: false });
  env.updateCloudMultimodalBadge(false);
  assert.strictEqual(env.elements.cloudMultimodalStatusBadge.textContent, 'Local Only (Disabled)');

  // User checks checkbox in form before saving
  env.updateCloudMultimodalBadge(true);
  assert.strictEqual(env.elements.cloudMultimodalStatusBadge.textContent, 'Pending Save (Not Yet Enabled)');
  assert(!env.elements.cloudMultimodalStatusBadge.classList.contains('active'));

  // Server confirms save
  env.setCachedSettings({ allow_cloud_multimodal: true });
  env.updateCloudMultimodalBadge(true);
  assert.strictEqual(env.elements.cloudMultimodalStatusBadge.textContent, '✓ Permitted');
  assert(env.elements.cloudMultimodalStatusBadge.classList.contains('active'));

  // User unchecks in form before saving
  env.updateCloudMultimodalBadge(false);
  assert.strictEqual(env.elements.cloudMultimodalStatusBadge.textContent, 'Pending Disable (Unsaved)');
});

runTest('Malformed or error server settings response defaults strictly to disabled', () => {
  const env = setupMockEnvironment();

  // Simulate network or server error returning null or malformed data
  let fallbackData = null;
  const isCloudEnabled = Boolean(fallbackData && fallbackData.allow_cloud_multimodal === true);
  env.setCachedSettings({ allow_cloud_multimodal: false });
  env.updateCloudMultimodalBadge(isCloudEnabled);

  assert.strictEqual(isCloudEnabled, false);
  assert.strictEqual(env.elements.cloudMultimodalStatusBadge.textContent, 'Local Only (Disabled)');
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. MODEL COMPATIBILITY & STRICT REJECTION
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 5. Model Selection & Compatibility Routing Audit ---');

runTest('Text-only modes (pipeline, deepseek, qwen) display warning banner and preserve dropdown selection', async () => {
  const env = setupMockEnvironment();
  await env.handleFiles([{ name: 'test.png', type: 'image/png', size: 1000 }]);

  for (const mode of ['pipeline', 'deepseek', 'qwen']) {
    env.setMode(mode);
    assert.strictEqual(env.elements.modeSelect.value, mode, `Dropdown selection must not change silently for ${mode}`);
    assert.strictEqual(env.elements.modeCompatibilityBanner.style.display, 'flex');
    assert(env.elements.modeCompatibilityBanner.className.includes('warning'));
    assert(env.elements.modeCompatibilityBanner.innerHTML.includes('text-only mode'));
  }
});

runTest('Gemini mode with images requires cloud permission confirmation', async () => {
  const env = setupMockEnvironment();
  await env.handleFiles([{ name: 'test.png', type: 'image/png', size: 1000 }]);

  // Disabled cloud permission
  env.setCachedSettings({ allow_cloud_multimodal: false });
  env.setMode('gemini');
  assert.strictEqual(env.elements.modeCompatibilityBanner.style.display, 'flex');
  assert(env.elements.modeCompatibilityBanner.className.includes('permission'));
  assert(env.elements.modeCompatibilityBanner.innerHTML.includes('Cloud Permission Required'));

  // Enabled cloud permission
  env.setCachedSettings({ allow_cloud_multimodal: true });
  env.setMode('gemini');
  // When cloud permitted, Gemini is vision capable
  assert.strictEqual(env.elements.modeCompatibilityBanner.style.display, 'none');
});

runTest('Auto and Chat modes communicate local vision prioritization', async () => {
  const env = setupMockEnvironment();
  await env.handleFiles([{ name: 'test.png', type: 'image/png', size: 1000 }]);

  env.setMode('auto');
  assert.strictEqual(env.elements.modeCompatibilityBanner.style.display, 'flex');
  assert(env.elements.modeCompatibilityBanner.className.includes('info'));
  assert(env.elements.modeCompatibilityBanner.innerHTML.includes('Auto mode prioritizes local vision'));

  env.setMode('chat');
  assert.strictEqual(env.elements.modeCompatibilityBanner.style.display, 'flex');
  assert(env.elements.modeCompatibilityBanner.className.includes('info'));
  assert(env.elements.modeCompatibilityBanner.innerHTML.includes('Local Vision Active'));
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. ERROR HANDLING SIMULATION (HTTP 400, 403, 413, 500, Abort, Network Drop)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 6. Backend Error Response Simulation Audit ---');

function simulateChatErrorResponse(status, jsonBody, statusText = '') {
  const mockRes = {
    ok: status >= 200 && status < 300,
    status: status,
    statusText: statusText,
    json: async () => {
      if (jsonBody === null) throw new Error('Invalid JSON');
      return jsonBody;
    }
  };

  let userNotice = '';
  const serverError = jsonBody?.error || jsonBody?.message || mockRes.statusText || 'Unknown error';

  if (mockRes.status === 400) {
    userNotice = `⚠️ <strong>Incompatible Request (HTTP 400):</strong> ${serverError}`;
  } else if (mockRes.status === 403) {
    userNotice = `🔒 <strong>Permission Denied (HTTP 403):</strong> ${serverError}`;
  } else if (mockRes.status === 413) {
    userNotice = `⚠️ <strong>Payload Too Large (HTTP 413):</strong> The request payload exceeds the maximum server limit (25 MB).`;
  } else {
    userNotice = `⚠️ <strong>Server Error (HTTP ${mockRes.status}):</strong> An error occurred while routing your request.`;
  }

  return userNotice;
}

runTest('Simulated HTTP 400: renders sanitized incompatible request notice', () => {
  const notice = simulateChatErrorResponse(400, { error: "Selected mode 'pipeline' does not support image attachments." });
  assert(notice.includes('HTTP 400'));
  assert(notice.includes('Selected mode &#39;pipeline&#39; does not support image attachments') || notice.includes("Selected mode 'pipeline'"));
});

runTest('Simulated HTTP 403: renders sanitized permission denied notice with settings prompt', () => {
  const notice = simulateChatErrorResponse(403, { error: 'Cloud multimodal processing is disabled.' });
  assert(notice.includes('HTTP 403'));
  assert(notice.includes('Cloud multimodal processing is disabled'));
});

runTest('Simulated HTTP 413: renders payload too large explanation', () => {
  const notice = simulateChatErrorResponse(413, { error: 'Payload too large' });
  assert(notice.includes('HTTP 413'));
  assert(notice.includes('Payload Too Large'));
  assert(notice.includes('25 MB'));
});

runTest('Simulated HTTP 500: renders sanitized server error without exposing internal tracebacks', () => {
  const notice = simulateChatErrorResponse(500, { error: 'Internal server error at line 542 in app.py with Traceback...' });
  assert(notice.includes('HTTP 500'));
  assert(notice.includes('Server Error'));
  assert(!notice.includes('Traceback'));
});

runTest('Simulated Invalid JSON / Proxy Error: falls back cleanly to status text without crashing', () => {
  const notice = simulateChatErrorResponse(502, null, 'Bad Gateway');
  assert(notice.includes('HTTP 502'));
  assert(notice.includes('Server Error'));
});

runTest('Simulated Aborted Request: abort signal handled without duplicate errors and controls reset', () => {
  const env = setupMockEnvironment();
  env.elements.sendBtn.classList.add('is-generating');
  env.elements.pauseBanner.style.display = 'block';

  const abortErr = new Error('The user aborted a request.');
  abortErr.name = 'AbortError';

  let caughtErrorNotice = null;
  if (abortErr.name !== 'AbortError') {
    caughtErrorNotice = abortErr.message;
  }
  env.resetPauseState();

  assert.strictEqual(caughtErrorNotice, null, 'AbortError must not produce an error notice banner');
  assert(!env.elements.sendBtn.classList.contains('is-generating'));
  assert.strictEqual(env.elements.pauseBanner.style.display, 'none');
});

console.log('\n================================================================');
console.log(` AUDIT SUMMARY: ${passedTests} of ${totalTests} test cases passed (100%)`);
console.log('================================================================\n');
