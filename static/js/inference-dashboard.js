(function () {
  const registry = window.InferenceSessions;
  if (!registry) {
    return;
  }

  const container = document.getElementById('inferenceSessionsContainer');
  const emptyState = document.getElementById('inferenceSessionsEmpty');
  const refreshBtn = document.getElementById('refreshSessions');
  const clearBtn = document.getElementById('clearSessions');
  const createLinks = document.querySelectorAll('.js-create-inference');

  const LAST_COUNTER_KEY = 'visionroi_inference_last_cam_index';

  function parseCamIndex(value) {
    if (!value) {
      return null;
    }
    const match = /cam(\d+)/i.exec(String(value));
    if (!match) {
      return null;
    }
    const num = parseInt(match[1], 10);
    return Number.isFinite(num) ? num : null;
  }

  function getMaxIndexFromSessions() {
    const sessions = registry.getAll();
    let maxIndex = 0;
    sessions.forEach((session) => {
      const candidates = [session?.id, session?.href];
      candidates.forEach((value) => {
        const parsed = parseCamIndex(value);
        if (parsed && parsed > maxIndex) {
          maxIndex = parsed;
        }
      });
    });
    return maxIndex;
  }

  function getNextCamId() {
    let stored = parseInt(localStorage.getItem(LAST_COUNTER_KEY), 10);
    if (!Number.isFinite(stored) || stored < 0) {
      stored = 0;
    }
    const maxExisting = getMaxIndexFromSessions();
    const nextIndex = Math.max(stored, maxExisting) + 1;
    try {
      localStorage.setItem(LAST_COUNTER_KEY, String(nextIndex));
    } catch (err) {
      console.warn('Failed to persist inference counter', err);
    }
    return `cam${nextIndex}`;
  }

  function buildTargetHref(baseHref, camId) {
    const href = typeof baseHref === 'string' && baseHref.length > 0 ? baseHref : '/inference';
    const hashIndex = href.indexOf('#');
    const cleanBase = hashIndex > -1 ? href.slice(0, hashIndex) : href;
    return `${cleanBase}#${camId}`;
  }

  createLinks.forEach((link) => {
    link.addEventListener('click', (event) => {
      if (event.button && event.button !== 0) {
        return;
      }
      event.preventDefault();
      const camId = getNextCamId();
      const target = buildTargetHref(link.getAttribute('href'), camId);
      window.location.href = target;
    });
  });

  const STATUS_BADGES = {
    running: { text: 'กำลังทำงาน', className: 'badge bg-success-subtle text-success' },
    stopped: { text: 'หยุดแล้ว', className: 'badge bg-secondary-subtle text-secondary' },
    error: { text: 'ผิดพลาด', className: 'badge bg-danger-subtle text-danger' },
  };

  function formatRelativeTime(timestamp) {
    if (!timestamp) {
      return '—';
    }
    const diff = Date.now() - timestamp;
    const seconds = Math.max(0, Math.floor(diff / 1000));
    if (seconds < 45) {
      return 'ไม่กี่วินาทีที่แล้ว';
    }
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) {
      return `${minutes} นาทีที่แล้ว`;
    }
    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
      return `${hours} ชั่วโมงที่แล้ว`;
    }
    const days = Math.floor(hours / 24);
    if (days < 7) {
      return `${days} วันที่แล้ว`;
    }
    const date = new Date(timestamp);
    return date.toLocaleString('th-TH', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  function createSessionElement(session) {
    const item = document.createElement('article');
    item.className = 'inference-session-item';

    const header = document.createElement('div');
    header.className = 'inference-session-item__header';

    const typeBadge = document.createElement('span');
    typeBadge.className = 'badge rounded-pill inference-session-item__type';
    typeBadge.textContent = session.type === 'page' ? 'Page Inference' : 'Group Inference';
    header.appendChild(typeBadge);

    const statusInfo = STATUS_BADGES[session.status] || STATUS_BADGES.running;
    const statusBadge = document.createElement('span');
    statusBadge.className = statusInfo.className + ' inference-session-item__status';
    statusBadge.textContent = statusInfo.text;
    header.appendChild(statusBadge);

    const title = document.createElement('h3');
    title.className = 'inference-session-item__title';
    title.textContent = session.title || 'Inference';

    const metaList = document.createElement('dl');
    metaList.className = 'inference-session-item__meta';

    function addMeta(label, value, icon) {
      if (!value) {
        return;
      }
      const dt = document.createElement('dt');
      dt.innerHTML = icon ? `<i class="${icon}"></i>${label}` : label;
      const dd = document.createElement('dd');
      dd.textContent = value;
      metaList.appendChild(dt);
      metaList.appendChild(dd);
    }

    if (session.sourceName || session.sourceId) {
      addMeta('Source', session.sourceName || session.sourceId, 'bi bi-camera-video me-1');
    }
    if (session.group) {
      addMeta('Group', session.group, 'bi bi-collection me-1');
    }
    if (session.page) {
      addMeta('Page', session.page, 'bi bi-file-earmark-text me-1');
    }

    const footer = document.createElement('div');
    footer.className = 'inference-session-item__footer';

    const timeInfo = document.createElement('div');
    timeInfo.className = 'inference-session-item__time text-muted';
    timeInfo.textContent = `อัปเดตล่าสุด ${formatRelativeTime(session.updatedAt)}`;

    const actions = document.createElement('div');
    actions.className = 'inference-session-item__actions';

    const openLink = document.createElement('a');
    openLink.className = 'btn btn-primary btn-sm';
    openLink.href = session.href || '/inference';
    openLink.innerHTML = '<i class="bi bi-box-arrow-up-right me-1"></i>เปิดหน้า';
    actions.appendChild(openLink);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn btn-outline-danger btn-sm';
    removeBtn.type = 'button';
    removeBtn.innerHTML = '<i class="bi bi-x-lg me-1"></i>นำออก';
    removeBtn.addEventListener('click', () => {
      const confirmed = window.confirm('ต้องการนำรายการนี้ออกจากหน้าสรุปหรือไม่?');
      if (confirmed) {
        registry.remove(session.id);
      }
    });
    actions.appendChild(removeBtn);

    item.appendChild(header);
    item.appendChild(title);
    item.appendChild(metaList);
    footer.appendChild(timeInfo);
    footer.appendChild(actions);
    item.appendChild(footer);

    return item;
  }

  function renderSessions(list) {
    container.innerHTML = '';
    if (!list || list.length === 0) {
      container.classList.add('is-empty');
      emptyState.classList.remove('d-none');
      return;
    }
    container.classList.remove('is-empty');
    emptyState.classList.add('d-none');

    list.forEach((session) => {
      const element = createSessionElement(session);
      container.appendChild(element);
    });
  }

  function refresh() {
    renderSessions(registry.getAll());
  }

  refreshBtn?.addEventListener('click', () => {
    refresh();
  });

  clearBtn?.addEventListener('click', () => {
    const confirmed = window.confirm('ต้องการล้างรายการทั้งหมดหรือไม่?');
    if (confirmed) {
      registry.clear();
    }
  });

  window.addEventListener('inferenceSessions:change', (event) => {
    const list = Array.isArray(event.detail) ? event.detail : registry.getAll();
    renderSessions(list);
  });

  window.addEventListener('storage', (event) => {
    if (event.key === registry.storageKey) {
      refresh();
    }
  });

  refresh();
})();
