(function (window) {
  const STORAGE_KEY = 'visionroi_inference_sessions';
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
    return Number.isFinite(num) && num > 0 ? num : null;
  }

  function updateLastCounterFromSession(session) {
    if (!session) {
      return;
    }
    const candidates = [session.id, session.href];
    let maxIndex = null;
    for (const value of candidates) {
      const parsed = parseCamIndex(value);
      if (parsed && (!maxIndex || parsed > maxIndex)) {
        maxIndex = parsed;
      }
    }
    if (!maxIndex) {
      return;
    }
    const current = parseInt(localStorage.getItem(LAST_COUNTER_KEY), 10);
    if (Number.isFinite(current) && current >= maxIndex) {
      return;
    }
    try {
      localStorage.setItem(LAST_COUNTER_KEY, String(maxIndex));
    } catch (err) {
      console.warn('Failed to persist inference counter', err);
    }
  }

  function resetLastCounter() {
    try {
      localStorage.removeItem(LAST_COUNTER_KEY);
    } catch (err) {
      console.warn('Failed to reset inference counter', err);
    }
  }

  function readSessions() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed;
    } catch (err) {
      console.warn('Failed to parse inference sessions from storage', err);
      return [];
    }
  }

  function writeSessions(list) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
    return list;
  }

  function emitChange(list) {
    window.dispatchEvent(
      new CustomEvent('inferenceSessions:change', {
        detail: list.slice(),
      })
    );
  }

  function normalizeSession(session) {
    const now = Date.now();
    return {
      id: String(session.id ?? ''),
      type: session.type === 'page' ? 'page' : 'group',
      title: session.title || 'Inference',
      sourceId: session.sourceId || '',
      sourceName: session.sourceName || '',
      group: session.group || '',
      page: session.page || '',
      href: session.href || '/inference',
      status: session.status || 'running',
      createdAt: session.createdAt || now,
      updatedAt: session.updatedAt || now,
      meta: session.meta || {},
    };
  }

  function sortSessions(list) {
    return list
      .slice()
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  }

  function upsertSession(data) {
    const sessions = readSessions();
    const normalized = normalizeSession(data);
    if (!normalized.id) {
      return;
    }
    const idx = sessions.findIndex((item) => item.id === normalized.id);
    if (idx > -1) {
      const merged = {
        ...sessions[idx],
        ...normalized,
        createdAt: sessions[idx].createdAt || normalized.createdAt,
        updatedAt: Date.now(),
      };
      sessions[idx] = merged;
    } else {
      sessions.push({ ...normalized, createdAt: Date.now(), updatedAt: Date.now() });
    }
    const finalList = writeSessions(sessions);
    updateLastCounterFromSession(normalized);
    emitChange(sortSessions(finalList));
  }

  function updateSession(id, updates) {
    if (!id) return;
    if (typeof updates !== 'object' || updates === null) {
      updates = {};
    }
    const sessions = readSessions();
    const idx = sessions.findIndex((item) => item.id === id);
    if (idx === -1) {
      return;
    }
    const now = Date.now();
    const merged = {
      ...sessions[idx],
      ...updates,
      updatedAt: now,
    };
    sessions[idx] = merged;
    const finalList = writeSessions(sessions);
    emitChange(sortSessions(finalList));
  }

  function removeSession(id) {
    if (!id) return;
    const sessions = readSessions();
    const next = sessions.filter((item) => item.id !== id);
    const finalList = writeSessions(next);
    emitChange(sortSessions(finalList));
  }

  function clearSessions() {
    localStorage.removeItem(STORAGE_KEY);
    resetLastCounter();
    emitChange([]);
  }

  function getAllSessions() {
    return sortSessions(readSessions());
  }

  window.InferenceSessions = {
    storageKey: STORAGE_KEY,
    getAll: getAllSessions,
    upsertSession,
    updateSession,
    remove: removeSession,
    clear: clearSessions,
  };
})(window);
