/**
 * DrugFX Frontend — app.js v2.0
 * Handles UI state, form submissions, loading steps, and result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── DOM refs ───────────────────────────────────────────────
  const btnText  = document.getElementById('btn-text');
  const btnImage = document.getElementById('btn-image');
  const formText  = document.getElementById('form-text');
  const formImage = document.getElementById('form-image');

  const uploadZone    = document.getElementById('upload-zone');
  const imageInput    = document.getElementById('image-input');
  const uploadContent = document.getElementById('upload-content');
  const uploadPreview = document.getElementById('upload-preview');
  const previewImg    = document.getElementById('preview-img');
  const previewName   = document.getElementById('preview-name');
  const previewClear  = document.getElementById('preview-clear');

  const btnAnalyzeText  = document.getElementById('btn-analyze-text');
  const btnAnalyzeImage = document.getElementById('btn-analyze-image');

  // Result panels
  const emptyState   = document.getElementById('empty-state');
  const loadingState = document.getElementById('loading-state');
  const resultsWrap  = document.getElementById('results-wrap');
  const errorState   = document.getElementById('error-state');
  const errorMsg     = document.getElementById('error-msg');
  const btnRetry     = document.getElementById('btn-retry');

  // Result sections
  const drugName           = document.getElementById('drug-name');
  const statusBadge        = document.getElementById('status-badge');
  const resultBadges       = document.getElementById('result-badges');
  const btnCopy            = document.getElementById('btn-copy');
  const sectionOcr         = document.getElementById('section-ocr');
  const ocrText            = document.getElementById('ocr-text');
  const sectionMeta        = document.getElementById('section-meta');
  const metaGrid           = document.getElementById('meta-grid');
  const synopsisText       = document.getElementById('synopsis-text');
  const keySeGrid          = document.getElementById('key-se-grid');
  const seList             = document.getElementById('se-list');
  const usesList           = document.getElementById('uses-list');
  const dosageText         = document.getElementById('dosage-text');
  const warnList           = document.getElementById('warn-list');
  const interactionsTags   = document.getElementById('interactions-tags');
  const alternativesTags   = document.getElementById('alternatives-tags');

  // Loading steps
  const ls1 = document.getElementById('ls-1');
  const ls2 = document.getElementById('ls-2');
  const ls3 = document.getElementById('ls-3');

  let lastResultData = null;
  let loadingTimer = null;

  // ── Mode Switcher ──────────────────────────────────────────
  btnText.addEventListener('click', () => switchMode('text'));
  btnImage.addEventListener('click', () => switchMode('image'));

  function switchMode(mode) {
    if (mode === 'text') {
      btnText.classList.add('active');
      btnText.setAttribute('aria-selected', 'true');
      btnImage.classList.remove('active');
      btnImage.setAttribute('aria-selected', 'false');
      formText.classList.remove('hidden');
      formText.classList.add('visible');
      formImage.classList.add('hidden');
      formImage.classList.remove('visible');
    } else {
      btnImage.classList.add('active');
      btnImage.setAttribute('aria-selected', 'true');
      btnText.classList.remove('active');
      btnText.setAttribute('aria-selected', 'false');
      formImage.classList.remove('hidden');
      formImage.classList.add('visible');
      formText.classList.add('hidden');
      formText.classList.remove('visible');
    }
  }

  // ── Image Upload ───────────────────────────────────────────
  imageInput.addEventListener('change', handleFileSelect);
  previewClear.addEventListener('click', clearFileSelect);

  ['dragover', 'dragleave', 'drop'].forEach(ev => {
    uploadZone.addEventListener(ev, e => e.preventDefault());
  });

  uploadZone.addEventListener('dragover', () => uploadZone.classList.add('dragover'));
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', (e) => {
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      imageInput.files = e.dataTransfer.files;
      handleFileSelect();
    }
  });

  function handleFileSelect() {
    const file = imageInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      uploadContent.classList.add('hidden');
      uploadPreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function clearFileSelect() {
    imageInput.value = '';
    previewImg.src = '#';
    uploadContent.classList.remove('hidden');
    uploadPreview.classList.add('hidden');
  }

  // ── UI State Management ────────────────────────────────────
  function showEmpty() {
    emptyState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    resultsWrap.classList.add('hidden');
    errorState.classList.add('hidden');
  }

  function showLoading() {
    emptyState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultsWrap.classList.add('hidden');
    errorState.classList.add('hidden');
    animateLoadingSteps();
  }

  function showResults() {
    emptyState.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultsWrap.classList.remove('hidden');
    errorState.classList.add('hidden');
    clearTimeout(loadingTimer);
  }

  function showError(msg) {
    emptyState.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultsWrap.classList.add('hidden');
    errorState.classList.remove('hidden');
    errorMsg.textContent = msg || 'An unexpected error occurred. Please try again.';
    clearTimeout(loadingTimer);
  }

  function animateLoadingSteps() {
    [ls1, ls2, ls3].forEach(el => { el.classList.remove('active', 'done'); });
    ls1.classList.add('active');
    loadingTimer = setTimeout(() => {
      ls1.classList.remove('active'); ls1.classList.add('done');
      ls2.classList.add('active');
      loadingTimer = setTimeout(() => {
        ls2.classList.remove('active'); ls2.classList.add('done');
        ls3.classList.add('active');
      }, 1800);
    }, 1000);
  }

  function setButtonLoading(btn, isLoading) {
    if (isLoading) {
      btn.classList.add('loading');
      btn.disabled = true;
    } else {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  }

  // ── Render Results ─────────────────────────────────────────
  function renderResults(response) {
    const data = response.data;
    if (!data) {
      showError(response.error || 'No data returned from the server.');
      return;
    }

    lastResultData = response;

    // Drug header
    drugName.textContent = data.drug_name || 'Unknown Drug';

    // Status badge
    resultBadges.innerHTML = `<span class="rbadge rbadge--success">✓ Analysis Complete</span>`;
    if (response.input_type === 'image') {
      resultBadges.innerHTML += `<span class="rbadge rbadge--image">📷 Image Scan</span>`;
    }

    // OCR extracted text
    if (response.input_type === 'image' && response.extracted_text) {
      sectionOcr.classList.remove('hidden');
      ocrText.textContent = response.extracted_text;
    } else {
      sectionOcr.classList.add('hidden');
    }

    // Label metadata (MFG / Expiry / Batch)
    const hasMfg     = data.mfg_date;
    const hasExp     = data.expiry_date;
    const hasBatch   = data.batch_no;

    if (response.input_type === 'image') {
      sectionMeta.classList.remove('hidden');
      metaGrid.innerHTML = buildMetaItem('Manufacturing Date', hasMfg)
        + buildMetaItem('Expiry Date', hasExp)
        + buildMetaItem('Batch / Lot No.', hasBatch);
    } else {
      sectionMeta.classList.add('hidden');
    }

    // Synopsis
    synopsisText.textContent = data.synopsis || 'No synopsis available.';

    // Critical Side Effects
    keySeGrid.innerHTML = '';
    const keyEffects = Array.isArray(data.key_side_effects) ? data.key_side_effects : [];
    if (keyEffects.length > 0) {
      keyEffects.forEach((eff, i) => {
        const chip = document.createElement('div');
        chip.className = 'kse-chip';
        chip.style.animationDelay = `${i * 60}ms`;
        chip.textContent = eff;
        keySeGrid.appendChild(chip);
      });
      document.getElementById('section-key-se').classList.remove('hidden');
    } else {
      document.getElementById('section-key-se').classList.add('hidden');
    }

    // Full Side Effects
    renderList(seList, data.side_effects, 'No side effects listed.');

    // Uses
    renderList(usesList, data.uses, 'No uses listed.');

    // Dosage
    dosageText.textContent = data.dosage || 'Consult your prescriber.';

    // Warnings
    renderWarnList(warnList, data.warnings);

    // Interactions
    renderTagCloud(interactionsTags, data.drug_interactions, 'tag--teal');

    // Alternatives
    renderTagCloud(alternativesTags, data.alternatives, 'tag--indigo');

    showResults();
  }

  function buildMetaItem(label, value) {
    if (value) {
      return `<div class="meta-item">
        <span class="meta-label">${label}</span>
        <span class="meta-value">${escHtml(value)}</span>
      </div>`;
    } else {
      return `<div class="meta-item">
        <span class="meta-label">${label}</span>
        <span class="meta-value not-detected">Not detected on label</span>
      </div>`;
    }
  }

  function renderList(ulEl, items, emptyMsg) {
    ulEl.innerHTML = '';
    const arr = Array.isArray(items) ? items : [];
    if (arr.length === 0) {
      const li = document.createElement('li');
      li.textContent = emptyMsg;
      li.style.color = 'var(--text-3)';
      li.style.fontStyle = 'italic';
      ulEl.appendChild(li);
    } else {
      arr.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        ulEl.appendChild(li);
      });
    }
  }

  function renderWarnList(ulEl, items) {
    ulEl.innerHTML = '';
    const arr = Array.isArray(items) ? items : [];
    if (arr.length === 0) {
      const li = document.createElement('li');
      li.textContent = 'Always consult a medical professional before use.';
      ulEl.appendChild(li);
    } else {
      arr.forEach(w => {
        const li = document.createElement('li');
        li.textContent = w;
        ulEl.appendChild(li);
      });
    }
  }

  function renderTagCloud(containerEl, items, cls) {
    containerEl.innerHTML = '';
    const arr = Array.isArray(items) ? items : [];
    if (arr.length === 0) {
      containerEl.innerHTML = `<span style="font-size:0.82rem;color:var(--text-3);font-style:italic;">None listed</span>`;
    } else {
      arr.forEach(item => {
        const tag = document.createElement('span');
        tag.className = `tag ${cls}`;
        tag.textContent = item;
        containerEl.appendChild(tag);
      });
    }
  }

  function escHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Copy Button ────────────────────────────────────────────
  btnCopy.addEventListener('click', () => {
    if (!lastResultData || !lastResultData.data) return;
    const d = lastResultData.data;
    const text = [
      `Drug: ${d.drug_name}`,
      `\nSynopsis:\n${d.synopsis}`,
      `\nUses:\n${(d.uses || []).map(u => '• ' + u).join('\n')}`,
      `\nSide Effects:\n${(d.side_effects || []).map(s => '• ' + s).join('\n')}`,
      `\nKey Side Effects:\n${(d.key_side_effects || []).map(k => '⚠ ' + k).join('\n')}`,
      `\nDosage:\n${d.dosage}`,
      `\nWarnings:\n${(d.warnings || []).map(w => '! ' + w).join('\n')}`,
      `\nDrug Interactions:\n${(d.drug_interactions || []).join(', ')}`,
      `\nAlternatives:\n${(d.alternatives || []).join(', ')}`,
      d.mfg_date ? `\nMFG Date: ${d.mfg_date}` : '',
      d.expiry_date ? `\nExpiry: ${d.expiry_date}` : '',
    ].filter(Boolean).join('');

    navigator.clipboard.writeText(text).then(() => {
      const orig = btnCopy.innerHTML;
      btnCopy.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>`;
      btnCopy.style.color = 'var(--green)';
      setTimeout(() => {
        btnCopy.innerHTML = orig;
        btnCopy.style.color = '';
      }, 2000);
    }).catch(() => {});
  });

  // ── Retry Button ───────────────────────────────────────────
  btnRetry.addEventListener('click', showEmpty);

  // ── Text Form Submit ───────────────────────────────────────
  formText.addEventListener('submit', async (e) => {
    e.preventDefault();
    const textVal = document.getElementById('text-input').value.trim();
    if (!textVal) return;

    setButtonLoading(btnAnalyzeText, true);
    showLoading();

    try {
      const fd = new FormData();
      fd.append('text', textVal);

      const res = await fetch('/analyze/text', { method: 'POST', body: fd });
      const json = await res.json();

      if (!res.ok) {
        throw new Error(json.detail || `Server error ${res.status}`);
      }
      if (!json.success) {
        throw new Error(json.error || 'Analysis returned failure status.');
      }

      renderResults(json);
    } catch (err) {
      showError(err.message);
    } finally {
      setButtonLoading(btnAnalyzeText, false);
    }
  });

  // ── Image Form Submit ──────────────────────────────────────
  formImage.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!imageInput.files.length) {
      alert('Please select an image first.');
      return;
    }

    setButtonLoading(btnAnalyzeImage, true);
    showLoading();

    try {
      const fd = new FormData();
      fd.append('file', imageInput.files[0]);

      const res = await fetch('/analyze/image', { method: 'POST', body: fd });
      const json = await res.json();

      if (!res.ok) {
        throw new Error(json.detail || `Server error ${res.status}`);
      }
      if (!json.success) {
        throw new Error(json.error || 'Could not extract text from the image.');
      }

      renderResults(json);
    } catch (err) {
      showError(err.message);
    } finally {
      setButtonLoading(btnAnalyzeImage, false);
    }
  });

  // ── Init ───────────────────────────────────────────────────
  showEmpty();
});
