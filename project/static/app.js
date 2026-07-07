/**
 * DrugFX Frontend Controller — app.js v3.0
 * Fully rewritten single-page application orchestrating views, theme management,
 * debounced search suggestions, drag-drop uploads, OCR review, loading milestones,
 * results dashboard, and multiple formats export.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ─── STATE MANAGEMENT ───────────────────────────────────────
  const state = {
    theme: localStorage.getItem('drugfx-theme') || 'dark',
    view: 'landing', // landing | ocr-review | loading | dashboard | error
    activeFile: null,
    ocrText: '',
    ocrConfidence: 0,
    ocrProvider: '',
    lastResult: null,
    debounceTimer: null,
    searchFocusedIndex: -1,
    suggestions: []
  };

  // ─── DOM SELECTORS ──────────────────────────────────────────
  // Layout views
  const viewLanding = document.getElementById('view-landing');
  const viewOcrReview = document.getElementById('view-ocr-review');
  const viewLoading = document.getElementById('view-loading');
  const viewDashboard = document.getElementById('view-dashboard');
  const viewError = document.getElementById('view-error');

  // Navigation & Theme
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const logoHomeLink = document.getElementById('logo-home-link');
  const navHome = document.getElementById('nav-home');
  const navHow = document.getElementById('nav-how');
  const navFeatures = document.getElementById('nav-features');
  const navFaq = document.getElementById('nav-faq');
  const footerHomeTriggers = document.querySelectorAll('.footer-nav-home-trigger');
  const btnHeaderAnalyze = document.getElementById('btn-header-analyze');

  // Search UI
  const searchInputField = document.getElementById('search-input-field');
  const suggestionsBox = document.getElementById('suggestions-box');
  const btnSearchSearch = document.getElementById('btn-search-search');
  const chipButtons = document.querySelectorAll('.example-searches .chip');

  // Upload UI
  const fileUploaderInput = document.getElementById('file-uploader-input');
  const dragDropZone = document.getElementById('drag-drop-zone');
  const btnUploadTriggerShortcut = document.getElementById('btn-upload-trigger-shortcut');
  const dragzoneIdleState = document.getElementById('dragzone-idle-state');
  const dragzoneSelectedState = document.getElementById('dragzone-selected-state');
  const previewThumbnailImg = document.getElementById('preview-thumbnail-img');
  const previewFilenameText = document.getElementById('preview-filename-text');
  const btnClearFile = document.getElementById('btn-clear-file');

  // OCR Review UI
  const reviewOriginalImg = document.getElementById('review-original-img');
  const reviewTextEditor = document.getElementById('review-text-editor');
  const ocrConfidenceText = document.getElementById('ocr-confidence-text');
  const ocrProviderText = document.getElementById('ocr-provider-text');
  const btnOcrCancel = document.getElementById('btn-ocr-cancel');
  const btnOcrProceed = document.getElementById('btn-ocr-proceed');

  // Loading Steps UI
  const stepOcr = document.getElementById('step-ocr');
  const stepRag = document.getElementById('step-rag');
  const stepLlm = document.getElementById('step-llm');

  // Dashboard UI
  const dashDrugName = document.getElementById('dash-drug-name');
  const dashGenericName = document.getElementById('dash-generic-name');
  const dashDrugClass = document.getElementById('dash-drug-class');
  const dashConfidenceBadge = document.getElementById('dash-confidence-badge');
  const dashSynopsis = document.getElementById('dash-synopsis');
  
  // Dashboard details elements
  const dashMetaMfg = document.getElementById('dash-meta-mfg');
  const dashMetaExp = document.getElementById('dash-meta-exp');
  const dashMetaBatch = document.getElementById('dash-meta-batch');
  const dashUsesList = document.getElementById('dash-uses-list');
  const dashDosageAdult = document.getElementById('dash-dosage-adult');
  const dashDosageMax = document.getElementById('dash-dosage-max');
  const dashDosageFood = document.getElementById('dash-dosage-food');
  const dashDosageDesc = document.getElementById('dash-dosage-desc');
  const dashCommonSeList = document.getElementById('dash-common-se-list');
  const dashSeriousSeList = document.getElementById('dash-serious-se-list');
  const dashWarningsList = document.getElementById('dash-warnings-list');
  const dashInteractionsTags = document.getElementById('dash-interactions-tags');
  const dashAlternativesTags = document.getElementById('dash-alternatives-tags');
  
  // Safety advisories elements
  const dashSafetyPregnancy = document.getElementById('dash-safety-pregnancy');
  const dashSafetyBreastfeeding = document.getElementById('dash-safety-breastfeeding');
  const dashSafetyAlcohol = document.getElementById('dash-safety-alcohol');
  const dashSafetyDriving = document.getElementById('dash-safety-driving');
  
  // Storage & Overdose
  const dashStorageVal = document.getElementById('dash-storage-val');
  const dashOverdoseVal = document.getElementById('dash-overdose-val');
  const dashMissedDoseVal = document.getElementById('dash-missed-dose-val');
  
  // FAQ Dashboard List
  const dashFaqList = document.getElementById('dash-faq-list');

  // Export buttons
  const btnExportCopy = document.getElementById('btn-export-copy');
  const btnExportPdf = document.getElementById('btn-export-pdf');
  const btnExportJson = document.getElementById('btn-export-json');
  const btnActionNewSearch = document.getElementById('btn-action-new-search');

  // Sidebar target buttons
  const sidebarMenuBtns = document.querySelectorAll('.sidebar-menu-btn');

  // Error Recovery UI
  const errorMessageText = document.getElementById('error-message-text');
  const btnErrorHome = document.getElementById('btn-error-home');
  const btnErrorRetry = document.getElementById('btn-error-retry');

  // ─── THEME CONFIGURATION ────────────────────────────────────
  function applyTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('drugfx-theme', state.theme);
  }

  themeToggleBtn.addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    applyTheme();
  });

  // Apply default configured theme
  applyTheme();

  // ─── VIEW ROUTER ────────────────────────────────────────────
  function showView(targetView) {
    state.view = targetView;
    
    // Hide all views
    [viewLanding, viewOcrReview, viewLoading, viewDashboard, viewError].forEach(v => {
      v.classList.add('hidden');
    });

    // Show active view
    if (targetView === 'landing') {
      viewLanding.classList.remove('hidden');
      resetNavActive(navHome);
    } else if (targetView === 'ocr-review') {
      viewOcrReview.classList.remove('hidden');
    } else if (targetView === 'loading') {
      viewLoading.classList.remove('hidden');
    } else if (targetView === 'dashboard') {
      viewDashboard.classList.remove('hidden');
    } else if (targetView === 'error') {
      viewError.classList.remove('hidden');
    }

    // Scroll to top on view changes
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function resetNavActive(activeEl) {
    [navHome, navHow, navFeatures, navFaq].forEach(l => l.classList.remove('active'));
    if (activeEl) activeEl.classList.add('active');
  }

  // Bind Navbar Home link
  logoHomeLink.addEventListener('click', (e) => {
    e.preventDefault();
    showView('landing');
  });
  navHome.addEventListener('click', (e) => {
    e.preventDefault();
    showView('landing');
  });
  btnHeaderAnalyze.addEventListener('click', () => {
    showView('landing');
    window.scrollTo({ top: dragDropZone.offsetTop - 120, behavior: 'smooth' });
  });

  // Smooth scroll links for standard navbar anchors
  navHow.addEventListener('click', (e) => {
    e.preventDefault();
    showView('landing');
    resetNavActive(navHow);
    document.getElementById('how-it-works-section').scrollIntoView({ behavior: 'smooth' });
  });
  navFeatures.addEventListener('click', (e) => {
    e.preventDefault();
    showView('landing');
    resetNavActive(navFeatures);
    document.getElementById('features-section').scrollIntoView({ behavior: 'smooth' });
  });
  navFaq.addEventListener('click', (e) => {
    e.preventDefault();
    showView('landing');
    resetNavActive(navFaq);
    document.getElementById('faq-section').scrollIntoView({ behavior: 'smooth' });
  });

  footerHomeTriggers.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      showView('landing');
    });
  });

  // Bind Sidebar Navigation inside Dashboard
  sidebarMenuBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sidebarMenuBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-target');
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Accordion toggle on FAQ page
  const faqItems = document.querySelectorAll('.faq-item');
  faqItems.forEach(item => {
    const questionBtn = item.querySelector('.faq-question');
    questionBtn.addEventListener('click', () => {
      const isActive = item.classList.contains('active');
      faqItems.forEach(i => i.classList.remove('active'));
      if (!isActive) {
        item.classList.add('active');
      }
    });
  });

  // ─── DEBOUNCED SEARCH & AUTOCOMPLETE ─────────────────────────
  searchInputField.addEventListener('input', () => {
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
      fetchSuggestions(searchInputField.value);
    }, 300);
  });

  async function fetchSuggestions(query) {
    if (!query || query.trim().length < 1) {
      hideSuggestions();
      return;
    }

    try {
      const response = await fetch(`/api/search/suggest?q=${encodeURIComponent(query.trim())}`);
      const data = await response.json();
      state.suggestions = data.suggestions || [];
      renderSuggestions();
    } catch (err) {
      console.error("Failed to fetch suggestions", err);
    }
  }

  function renderSuggestions() {
    if (state.suggestions.length === 0) {
      hideSuggestions();
      return;
    }

    suggestionsBox.innerHTML = '';
    state.suggestions.forEach((suggest, idx) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = suggest;
      div.setAttribute('role', 'option');
      div.setAttribute('id', `suggestion-opt-${idx}`);
      
      div.addEventListener('click', () => {
        searchInputField.value = suggest;
        hideSuggestions();
        triggerTextSearch(suggest);
      });
      suggestionsBox.appendChild(div);
    });

    suggestionsBox.classList.add('active');
    state.searchFocusedIndex = -1;
  }

  function hideSuggestions() {
    suggestionsBox.classList.remove('active');
    state.searchFocusedIndex = -1;
  }

  // Keyboard navigation for suggestions
  searchInputField.addEventListener('keydown', (e) => {
    const items = suggestionsBox.querySelectorAll('.suggestion-item');
    if (!suggestionsBox.classList.contains('active') || items.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      state.searchFocusedIndex = (state.searchFocusedIndex + 1) % items.length;
      updateSuggestionFocus(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      state.searchFocusedIndex = (state.searchFocusedIndex - 1 + items.length) % items.length;
      updateSuggestionFocus(items);
    } else if (e.key === 'Enter') {
      if (state.searchFocusedIndex > -1) {
        e.preventDefault();
        items[state.searchFocusedIndex].click();
      }
    } else if (e.key === 'Escape') {
      hideSuggestions();
    }
  });

  function updateSuggestionFocus(items) {
    items.forEach((item, idx) => {
      if (idx === state.searchFocusedIndex) {
        item.classList.add('focused');
        searchInputField.setAttribute('aria-activedescendant', item.id);
      } else {
        item.classList.remove('focused');
      }
    });
  }

  // Hide suggestions when clicking outside
  document.addEventListener('click', (e) => {
    if (!searchInputField.contains(e.target) && !suggestionsBox.contains(e.target)) {
      hideSuggestions();
    }
  });

  // Bind Search Trigger Button
  btnSearchSearch.addEventListener('click', () => {
    const val = searchInputField.value.trim();
    if (val) triggerTextSearch(val);
  });

  // Bind Chip example triggers
  chipButtons.forEach(chip => {
    chip.addEventListener('click', () => {
      const drug = chip.getAttribute('data-query');
      searchInputField.value = drug;
      triggerTextSearch(drug);
    });
  });

  // ─── FILE UPLOAD PROCESSING ────────────────────────────────
  // Direct file click triggers
  btnUploadTriggerShortcut.addEventListener('click', () => {
    fileUploaderInput.click();
  });
  dragDropZone.addEventListener('click', () => {
    fileUploaderInput.click();
  });
  
  // Support drag keyboard space/enter press trigger
  dragDropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileUploaderInput.click();
    }
  });

  // Handle Drag Over events
  ['dragenter', 'dragover'].forEach(eventName => {
    dragDropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dragDropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dragDropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dragDropZone.classList.remove('dragover');
    }, false);
  });

  // Handle Drop file select
  dragDropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileUploaderInput.addEventListener('change', () => {
    if (fileUploaderInput.files.length > 0) {
      handleFileSelection(fileUploaderInput.files[0]);
    }
  });

  function handleFileSelection(file) {
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      alert("File size exceeds 10MB limit. Please select a smaller file.");
      return;
    }
    state.activeFile = file;
    
    // Render Selected UI Preview
    previewFilenameText.textContent = file.name;
    
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewThumbnailImg.src = e.target.result;
        reviewOriginalImg.src = e.target.result;
      };
      reader.readAsDataURL(file);
    } else if (file.type === 'application/pdf') {
      previewThumbnailImg.src = 'https://img.icons8.com/color/96/pdf.png';
      reviewOriginalImg.src = 'https://img.icons8.com/color/96/pdf.png';
    }

    dragzoneIdleState.style.display = 'none';
    dragzoneSelectedState.style.display = 'flex';

    // Auto-trigger analysis for premium speed
    triggerImageOCRUpload(file);
  }

  btnClearFile.addEventListener('click', (e) => {
    e.stopPropagation(); // Avoid triggering open browse dialog
    clearUploadedFile();
  });

  function clearUploadedFile() {
    state.activeFile = null;
    fileUploaderInput.value = '';
    previewThumbnailImg.src = '';
    reviewOriginalImg.src = '';
    dragzoneSelectedState.style.display = 'none';
    dragzoneIdleState.style.display = 'flex';
  }

  // ─── PIPELINE TRIGGERS & ACTIONS ────────────────────────────
  
  // OCR Correction cancel
  btnOcrCancel.addEventListener('click', () => {
    clearUploadedFile();
    showView('landing');
  });

  // Submit corrected text for RAG+LLM analysis
  btnOcrProceed.addEventListener('click', () => {
    const correctedText = reviewTextEditor.value.trim();
    if (!correctedText) {
      alert("Verification text cannot be empty.");
      return;
    }
    triggerFinalAnalysis(correctedText);
  });

  // 1. Image Upload -> OCR Pipeline
  async function triggerImageOCRUpload(file) {
    showView('loading');
    updateLoadingStep('ocr', 'active');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/analyze/image', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || `Server status ${res.status}`);
      }

      if (data.success === false) {
        // Fallback to error view
        throw new Error(data.error?.message || data.error || 'Failed to parse image label.');
      }

      // Record OCR results in state
      state.ocrText = data.extracted_text || '';
      state.ocrConfidence = Math.round((data.ocr?.confidence || 0.85) * 100);
      state.ocrProvider = data.ocr?.provider || 'Gemini Vision';

      // Load OCR values into Review Text Area
      reviewTextEditor.value = state.ocrText;
      ocrConfidenceText.textContent = `${state.ocrConfidence}% Confidence`;
      ocrProviderText.textContent = state.ocrProvider === 'tesseract' ? 'Tesseract OCR' : 'Gemini Vision';

      // Switch to OCR Review view instead of direct dashboard rendering
      updateLoadingStep('ocr', 'done');
      showView('ocr-review');
    } catch (err) {
      console.error(err);
      triggerErrorState(err.message || 'The server encountered an error processing the image. Please verify you uploaded a valid label image.');
    }
  }

  // 2. Text input analysis
  async function triggerTextSearch(query) {
    showView('loading');
    updateLoadingStep('ocr', 'done'); // Skip OCR step since it's direct query
    updateLoadingStep('rag', 'active');

    try {
      const formData = new FormData();
      formData.append('text', query);

      const res = await fetch('/api/analyze/text', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `Server status ${res.status}`);
      }

      updateLoadingStep('rag', 'done');
      updateLoadingStep('llm', 'active');

      if (data.success === false) {
        throw new Error(data.error?.message || data.error || 'Medical database analysis failed.');
      }

      state.lastResult = data;
      renderDashboardReport(data.data);
      updateLoadingStep('llm', 'done');
      
      setTimeout(() => {
        showView('dashboard');
      }, 500);

    } catch (err) {
      console.error(err);
      triggerErrorState(err.message || 'Failure performing drug database lookup.');
    }
  }

  // 3. Final Analysis from Verified OCR editor
  async function triggerFinalAnalysis(text) {
    showView('loading');
    updateLoadingStep('ocr', 'done');
    updateLoadingStep('rag', 'active');

    try {
      const formData = new FormData();
      formData.append('text', text);

      const res = await fetch('/api/analyze/text', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `Server status ${res.status}`);
      }

      updateLoadingStep('rag', 'done');
      updateLoadingStep('llm', 'active');

      if (data.success === false) {
        throw new Error(data.error?.message || data.error || 'Analysis verification failed.');
      }

      state.lastResult = data;
      renderDashboardReport(data.data);
      updateLoadingStep('llm', 'done');
      
      setTimeout(() => {
        showView('dashboard');
      }, 500);

    } catch (err) {
      console.error(err);
      triggerErrorState(err.message || 'Failure verifying OCR text coordinates.');
    }
  }

  // Manage loading step visual updates
  function updateLoadingStep(step, status) {
    let element;
    if (step === 'ocr') element = stepOcr;
    else if (step === 'rag') element = stepRag;
    else if (step === 'llm') element = stepLlm;

    if (!element) return;

    const checkIcon = element.querySelector('.check-icon');

    if (status === 'active') {
      element.classList.add('active');
      element.classList.remove('done');
      if (checkIcon) checkIcon.style.display = 'none';
    } else if (status === 'done') {
      element.classList.add('done');
      element.classList.remove('active');
      if (checkIcon) checkIcon.style.display = 'block';
    }
  }

  // ─── RENDER MEDICAL DASHBOARD ───────────────────────────────
  function renderDashboardReport(data) {
    if (!data) return;

    // Header Title Elements
    dashDrugName.textContent = data.drug_name || 'Unknown';
    dashGenericName.textContent = data.generic_name ? `(${data.generic_name})` : '';
    dashDrugClass.textContent = `Class: ${data.drug_class || 'General Medicine'}`;
    
    // Synopsis
    dashSynopsis.textContent = data.synopsis || 'No drug synopsis available.';

    // Confidence Level badge styling
    const level = data.confidence?.level || 'medium';
    const score = Math.round((data.confidence?.score || 0.6) * 100);
    dashConfidenceBadge.className = `confidence-box ${level}`;
    dashConfidenceBadge.querySelector('span').textContent = `${level.toUpperCase()} CONFIDENCE (${score}%)`;

    // Label coordinates metadata fields
    dashMetaMfg.textContent = data.mfg_date || 'Not found';
    dashMetaExp.textContent = data.expiry_date || 'Not found';
    dashMetaBatch.textContent = data.batch_no || 'Not found';

    // Highlight missing label coordinates
    [dashMetaMfg, dashMetaExp, dashMetaBatch].forEach(field => {
      if (field.textContent === 'Not found') {
        field.classList.add('missing');
      } else {
        field.classList.remove('missing');
      }
    });

    // Uses list
    renderBulletList(dashUsesList, data.primary_uses, 'uses');

    // Dosage & Intake Values
    dashDosageAdult.textContent = data.dosage?.adult || 'As directed by physician';
    dashDosageMax.textContent = data.dosage?.max_dose || 'Not specified';
    
    const food = data.dosage?.with_food;
    dashDosageFood.textContent = food === true ? 'Take with food' : (food === false ? 'On empty stomach' : 'With or without food');
    
    // Handle nested dosage structures
    let dosageDetailsText = data.dosage;
    if (typeof data.dosage === 'object') {
      dosageDetailsText = `Frequency: ${data.dosage.frequency || 'N/A'}. Administration: ${data.administration || 'N/A'}. adjustments: ${data.dosage.elderly || 'N/A'}`;
    }
    dashDosageDesc.textContent = dosageDetailsText || 'Consult drug labels packaging for complete details.';

    // Side Effects Split Lists (Common vs Serious)
    renderBulletList(dashCommonSeList, data.common_side_effects, 'common-se');
    renderBulletList(dashSeriousSeList, data.serious_side_effects, 'serious-se');

    // Warnings list
    renderBulletList(dashWarningsList, data.warnings, 'warnings');

    // Interactions flex-tags cloud
    renderTagBadges(dashInteractionsTags, data.drug_interactions, 'teal');

    // Alternatives flex-tags cloud
    renderTagBadges(dashAlternativesTags, data.alternatives, 'indigo');

    // Safety Advisories
    dashSafetyPregnancy.textContent = data.pregnancy_safety || 'Category Unknown';
    dashSafetyBreastfeeding.textContent = data.breastfeeding_safety || 'Consult pediatrician';
    dashSafetyAlcohol.textContent = data.alcohol_interaction || 'Avoid alcohol';
    dashSafetyDriving.textContent = data.driving_advisory || 'Caution advised';

    // Storage, Missed dose & Overdose values
    dashStorageVal.textContent = data.storage || 'Store in cool, dry place.';
    dashOverdoseVal.textContent = data.overdose_guidance || 'Seek immediate medical attention.';
    dashMissedDoseVal.textContent = data.missed_dose || 'Take as soon as possible unless next dose is close.';

    // FAQ Cards Grid
    renderFAQGrid(dashFaqList, data.faq);
  }

  // Bullet items renderer
  function renderBulletList(ulElement, listData, type) {
    ulElement.innerHTML = '';
    const items = Array.isArray(listData) ? listData : [];
    
    if (items.length === 0) {
      const li = document.createElement('li');
      li.innerHTML = `<span>No recorded entries.</span>`;
      ulElement.appendChild(li);
      return;
    }

    items.forEach(itemText => {
      const li = document.createElement('li');
      
      let svgMarkup = '';
      if (type === 'uses') {
        svgMarkup = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
      } else if (type === 'common-se' || type === 'warnings') {
        svgMarkup = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`;
      } else if (type === 'serious-se') {
        svgMarkup = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
      }

      li.innerHTML = `${svgMarkup}<span>${escapeHtml(itemText)}</span>`;
      ulElement.appendChild(li);
    });
  }

  // Badge tags renderer
  function renderTagBadges(container, listData, colorClass) {
    container.innerHTML = '';
    const items = Array.isArray(listData) ? listData : [];

    if (items.length === 0) {
      container.innerHTML = `<span style="font-size: 0.85rem; font-style: italic; color: var(--text-muted);">None reported</span>`;
      return;
    }

    items.forEach(itemText => {
      const badge = document.createElement('span');
      badge.className = `tag-badge ${colorClass}`;
      badge.textContent = itemText;
      container.appendChild(badge);
    });
  }

  // FAQ dashboard grid renderer
  function renderFAQGrid(container, faqData) {
    container.innerHTML = '';
    const items = Array.isArray(faqData) ? faqData : [];

    if (items.length === 0) {
      container.innerHTML = `<div class="faq-card-item" style="color: var(--text-muted); font-style: italic;">No FAQs available for this medication.</div>`;
      return;
    }

    items.forEach(faq => {
      const div = document.createElement('div');
      div.className = 'faq-card-item';
      div.innerHTML = `
        <div class="fci-question">${escapeHtml(faq.q)}</div>
        <div class="fci-answer">${escapeHtml(faq.a)}</div>
      `;
      container.appendChild(div);
    });
  }

  function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
         .toString()
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
  }

  // ─── ERROR HANDLING RECOVERY ──────────────────────────────
  function triggerErrorState(message) {
    errorMessageText.textContent = message || 'We failed to analyze this medicine due to a server connection timeout.';
    showView('error');
  }

  btnErrorHome.addEventListener('click', () => {
    clearUploadedFile();
    showView('landing');
  });

  btnErrorRetry.addEventListener('click', () => {
    if (state.activeFile) {
      triggerImageOCRUpload(state.activeFile);
    } else {
      const val = searchInputField.value.trim();
      if (val) triggerTextSearch(val);
      else showView('landing');
    }
  });

  btnActionNewSearch.addEventListener('click', () => {
    clearUploadedFile();
    searchInputField.value = '';
    showView('landing');
  });

  // ─── EXPORT CHANNELS ────────────────────────────────────────

  // Copy structured report to clipboard
  btnExportCopy.addEventListener('click', () => {
    if (!state.lastResult || !state.lastResult.data) return;
    
    const d = state.lastResult.data;
    const textReport = [
      `DRUG REPORT: ${d.drug_name || 'N/A'} (${d.generic_name || 'N/A'})`,
      `Class: ${d.drug_class || 'N/A'}`,
      `Confidence: ${d.confidence?.level?.toUpperCase()} (${Math.round((d.confidence?.score || 0.6) * 100)}%)`,
      `\n[Synopsis]\n${d.synopsis || 'N/A'}`,
      `\n[Uses]\n${(d.primary_uses || []).map(u => '• ' + u).join('\n')}`,
      `\n[Dosage]\nAdult: ${d.dosage?.adult || 'N/A'}\nMax Daily: ${d.dosage?.max_dose || 'N/A'}\nFood: ${d.dosage?.with_food ? 'With Food' : 'With or without food'}`,
      `\n[Warnings]\n${(d.warnings || []).map(w => '⚠ ' + w).join('\n')}`,
      `\n[Common Side Effects]\n${(d.common_side_effects || []).map(s => '• ' + s).join('\n')}`,
      `\n[Serious Side Effects]\n${(d.serious_side_effects || []).map(s => '⚠ ' + s).join('\n')}`,
      `\n[Interactions]\n${(d.drug_interactions || []).join(', ')}`,
      `\n[Alternatives]\n${(d.alternatives || []).join(', ')}`
    ].join('\n');

    navigator.clipboard.writeText(textReport).then(() => {
      const origText = btnExportCopy.innerHTML;
      btnExportCopy.innerHTML = '<span>Copied!</span>';
      btnExportCopy.style.color = 'var(--success)';
      setTimeout(() => {
        btnExportCopy.innerHTML = origText;
        btnExportCopy.style.color = '';
      }, 2000);
    }).catch(err => {
      console.error("Clipboard copy failed", err);
    });
  });

  // Export as PDF via system window printing
  btnExportPdf.addEventListener('click', () => {
    window.print();
  });

  // Download raw JSON structure
  btnExportJson.addEventListener('click', () => {
    if (!state.lastResult) return;
    
    const jsonStr = JSON.stringify(state.lastResult, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    
    const drugNameSafe = (state.lastResult.data?.drug_name || 'drug_report')
      .toLowerCase()
      .replace(/\s+/g, '_');
      
    link.download = `drugfx_${drugNameSafe}.json`;
    document.body.appendChild(link);
    link.click();
    
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  });

});
