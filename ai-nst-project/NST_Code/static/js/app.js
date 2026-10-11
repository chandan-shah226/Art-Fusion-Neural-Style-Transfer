'use strict';

/* =======================================================
   CanvasFlow — app.js
   Handles: drag-drop uploads, alpha slider, form submit
   loading states, client-side validation
   ======================================================= */

/* ---- Alpha Slider Sync ---- */
(function initSlider() {
  var range   = document.getElementById('alphaRange');
  var display = document.getElementById('alphaDisplay');
  if (!range || !display) return;

  function sync() {
    display.textContent = parseFloat(range.value).toFixed(2);
  }

  range.addEventListener('input', sync);
  sync(); // initialise from server-rendered value
}());

/* ---- Drag-and-Drop File Upload ---- */
(function initUploads() {
  var ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
  var ALLOWED_EXT   = /\.(png|jpg|jpeg)$/i;

  function setup(dropId, inputId) {
    var drop  = document.getElementById(dropId);
    var input = document.getElementById(inputId);
    if (!drop || !input) return;

    /* --- Click drop zone → open file browser --- */
    drop.addEventListener('click', function (e) {
      if (e.target.closest('.preview-change')) return; // handled separately
      input.click();
    });

    /* --- Keyboard accessibility --- */
    drop.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        input.click();
      }
    });

    /* --- Drag events --- */
    drop.addEventListener('dragenter', function (e) {
      e.preventDefault();
      drop.classList.add('drag-over');
    });

    drop.addEventListener('dragover', function (e) {
      e.preventDefault();
      drop.classList.add('drag-over');
    });

    drop.addEventListener('dragleave', function (e) {
      /* Only remove class when cursor truly leaves the zone */
      if (!drop.contains(e.relatedTarget)) {
        drop.classList.remove('drag-over');
      }
    });

    drop.addEventListener('drop', function (e) {
      e.preventDefault();
      drop.classList.remove('drag-over');

      var file = e.dataTransfer.files[0];
      if (!file) return;

      if (!isAllowed(file)) {
        showError(drop, 'Unsupported format. Please use PNG or JPG.');
        return;
      }

      assignToInput(input, file);
      renderPreview(drop, file);
    });

    /* --- File-input change (browser dialog) --- */
    input.addEventListener('change', function () {
      var file = input.files[0];
      if (file) renderPreview(drop, file);
    });

    /* --- "Change" button inside preview --- */
    drop.addEventListener('click', function (e) {
      var btn = e.target.closest('.preview-change');
      if (btn) {
        e.stopPropagation();
        input.click();
      }
    });
  }

  /* Validate file type via MIME or extension */
  function isAllowed(file) {
    if (ALLOWED_TYPES.indexOf(file.type) !== -1) return true;
    return ALLOWED_EXT.test(file.name);
  }

  /* Use DataTransfer API to programmatically set files on the input */
  function assignToInput(input, file) {
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch (err) {
      /* Older browsers — file will be empty but drag-drop still shows preview */
    }
  }

  /* Render image preview inside the drop zone */
  function renderPreview(drop, file) {
    var reader = new FileReader();
    reader.onload = function (e) {
      var src  = e.target.result;
      var name = truncateName(file.name, 26);
      drop.innerHTML =
        '<div class="drop-preview">' +
          '<img class="preview-img" src="' + src + '" alt="Preview">' +
          '<div class="preview-meta">' +
            '<span class="preview-filename">' + escapeHTML(name) + '</span>' +
            '<button type="button" class="preview-change">Change</button>' +
          '</div>' +
        '</div>';
    };
    reader.readAsDataURL(file);
  }

  /* Show a validation error inside the idle drop zone */
  function showError(drop, msg) {
    var idle = drop.querySelector('.drop-idle');
    if (!idle) return;
    var existing = idle.querySelector('.drop-error');
    if (existing) existing.remove();
    var el = document.createElement('p');
    el.className = 'drop-error';
    el.textContent = msg;
    idle.appendChild(el);
  }

  /* Shorten long filenames for display */
  function truncateName(name, max) {
    if (name.length <= max) return name;
    var dot  = name.lastIndexOf('.');
    var ext  = dot !== -1 ? name.slice(dot) : '';
    var base = dot !== -1 ? name.slice(0, dot) : name;
    return base.slice(0, max - ext.length - 3) + '...' + ext;
  }

  /* Basic HTML escaping to avoid XSS in innerHTML */
  function escapeHTML(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  setup('contentDrop', 'contentInput');
}());

/* ---- Style Preset Selection ---- */
(function initStylePresets() {
  var items = document.querySelectorAll('.style-preset-item');
  var hiddenInput = document.querySelector('[name="style_path"]');
  if (!items.length || !hiddenInput) return;

  items.forEach(function(item) {
    item.addEventListener('click', function() {
      // Remove selected from all
      items.forEach(function(el) { el.classList.remove('selected'); });
      // Add selected to clicked
      item.classList.add('selected');
      // Update hidden input
      hiddenInput.value = item.getAttribute('data-style');
    });

    item.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        item.click();
      }
    });
  });
}());

/* ---- Form Submit: Validation + Loading Overlay ---- */
(function initFormSubmit() {
  var form    = document.getElementById('styleForm');
  var overlay = document.getElementById('loadingOverlay');
  var stepEl  = document.getElementById('loadingStep');
  if (!form || !overlay) return;

  var steps = [
    'Processing image\u2026',
    'Applying style\u2026',
    'Generating output\u2026'
  ];

  form.addEventListener('submit', function (e) {
    var contentInput = document.getElementById('contentInput');
    var contentPath  = form.querySelector('[name="content_path"]');
    var stylePath    = form.querySelector('[name="style_path"]');

    var hasContent = (contentInput && contentInput.files.length > 0) ||
                     (contentPath  && contentPath.value.trim() !== '');
    var hasStyle   = (stylePath    && stylePath.value.trim()    !== '');

    if (!hasContent || !hasStyle) {
      e.preventDefault();
      showClientError(form,
        !hasContent
          ? 'Please upload a content image before generating.'
          : 'Please upload a style reference image before generating.'
      );
      return;
    }

    removeClientError();

    /* Show loading overlay and cycle messages */
    overlay.classList.add('active');
    var idx      = 0;
    stepEl.textContent = steps[0];
    var interval = setInterval(function () {
      idx = (idx + 1) % steps.length;
      stepEl.textContent = steps[idx];
    }, 1600);

    window.addEventListener('pagehide', function () { clearInterval(interval); });
  });

  function showClientError(form, msg) {
    removeClientError();
    var el = document.createElement('div');
    el.id = 'cfClientError';
    el.className = 'alert alert-error';
    el.setAttribute('role', 'alert');
    el.innerHTML = '<i class="fas fa-exclamation-circle" aria-hidden="true"></i><span>' + msg + '</span>';
    form.insertAdjacentElement('afterend', el);
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function removeClientError() {
    var el = document.getElementById('cfClientError');
    if (el) el.remove();
  }
}());

/* ---- Scroll to Result on Page Load ---- */
(function scrollToResult() {
  var result = document.getElementById('resultSection');
  if (!result) return;
  setTimeout(function () {
    result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 150);
}());
