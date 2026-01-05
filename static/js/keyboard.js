/**
 * CallWhisper Keyboard Shortcuts
 *
 * Based on LibV2 accessibility-in-digital-design patterns:
 * - Keyboard navigation for accessibility
 * - Common shortcuts for power users
 * - Visual feedback for shortcut actions
 *
 * Shortcuts:
 *   Ctrl/Cmd + Space  - Toggle recording (start/stop)
 *   Ctrl/Cmd + S      - Save current transcription
 *   Ctrl/Cmd + R      - Refresh device list
 *   Ctrl/Cmd + K      - Focus search/filter
 *   Escape            - Cancel current operation / close modals
 *   ?                 - Show keyboard shortcuts help
 */

const KeyboardShortcuts = {
    // Registered shortcuts
    _shortcuts: {},

    // Whether shortcuts are enabled
    _enabled: true,

    // Elements that should block shortcuts (inputs, textareas, etc.)
    _blockingElements: ['INPUT', 'TEXTAREA', 'SELECT'],

    /**
     * Initialize keyboard shortcuts
     */
    init() {
        document.addEventListener('keydown', this._handleKeydown.bind(this));

        // Register default shortcuts
        this._registerDefaults();

        console.log('[Keyboard] Shortcuts initialized. Press ? for help.');
    },

    /**
     * Register default shortcuts
     */
    _registerDefaults() {
        // Toggle recording: Ctrl/Cmd + Space
        this.register('ctrl+space', {
            description: 'Toggle recording (start/stop)',
            category: 'Recording',
            handler: () => this._toggleRecording(),
        });

        // Save transcription: Ctrl/Cmd + S
        this.register('ctrl+s', {
            description: 'Save current transcription',
            category: 'File',
            handler: (e) => {
                e.preventDefault();
                this._saveTranscription();
            },
        });

        // Refresh devices: Ctrl/Cmd + R
        this.register('ctrl+r', {
            description: 'Refresh device list',
            category: 'Devices',
            handler: (e) => {
                e.preventDefault();
                this._refreshDevices();
            },
        });

        // Focus search: Ctrl/Cmd + K
        this.register('ctrl+k', {
            description: 'Focus search/filter',
            category: 'Navigation',
            handler: (e) => {
                e.preventDefault();
                this._focusSearch();
            },
        });

        // Cancel/Close: Escape
        this.register('escape', {
            description: 'Cancel operation / Close modal',
            category: 'General',
            handler: () => this._handleEscape(),
        });

        // Show help: ?
        this.register('?', {
            description: 'Show keyboard shortcuts',
            category: 'Help',
            allowInInput: false,
            handler: () => this._showHelp(),
        });

        // Reset: Ctrl + Shift + R
        this.register('ctrl+shift+r', {
            description: 'Reset to idle state',
            category: 'Recording',
            handler: (e) => {
                e.preventDefault();
                this._resetApp();
            },
        });
    },

    /**
     * Register a keyboard shortcut
     *
     * @param {string} combo - Key combination (e.g., 'ctrl+s', 'escape')
     * @param {Object} options - Shortcut options
     */
    register(combo, options) {
        const normalizedCombo = this._normalizeCombo(combo);
        this._shortcuts[normalizedCombo] = {
            combo: normalizedCombo,
            description: options.description || '',
            category: options.category || 'General',
            handler: options.handler,
            allowInInput: options.allowInInput !== false,
        };
    },

    /**
     * Unregister a shortcut
     */
    unregister(combo) {
        const normalizedCombo = this._normalizeCombo(combo);
        delete this._shortcuts[normalizedCombo];
    },

    /**
     * Enable/disable all shortcuts
     */
    setEnabled(enabled) {
        this._enabled = enabled;
    },

    /**
     * Normalize key combination string
     */
    _normalizeCombo(combo) {
        return combo
            .toLowerCase()
            .split('+')
            .sort((a, b) => {
                // Sort modifiers first: ctrl, alt, shift, then key
                const order = { ctrl: 0, cmd: 0, alt: 1, shift: 2 };
                const aOrder = order[a] ?? 3;
                const bOrder = order[b] ?? 3;
                return aOrder - bOrder;
            })
            .join('+');
    },

    /**
     * Build combo string from event
     */
    _getComboFromEvent(event) {
        const parts = [];

        if (event.ctrlKey || event.metaKey) {
            parts.push('ctrl');
        }
        if (event.altKey) {
            parts.push('alt');
        }
        if (event.shiftKey) {
            parts.push('shift');
        }

        // Get the key
        let key = event.key.toLowerCase();

        // Handle special keys
        if (key === ' ') {
            key = 'space';
        } else if (key === 'escape') {
            key = 'escape';
        }

        // Don't add modifier keys as the main key
        if (!['control', 'alt', 'shift', 'meta'].includes(key)) {
            parts.push(key);
        }

        return this._normalizeCombo(parts.join('+'));
    },

    /**
     * Handle keydown events
     */
    _handleKeydown(event) {
        if (!this._enabled) {
            return;
        }

        const combo = this._getComboFromEvent(event);
        const shortcut = this._shortcuts[combo];

        if (!shortcut) {
            return;
        }

        // Check if we should block in input elements
        const target = event.target;
        const isInputElement = this._blockingElements.includes(target.tagName);
        const isContentEditable = target.isContentEditable;

        if ((isInputElement || isContentEditable) && !shortcut.allowInInput) {
            // Allow escape in inputs
            if (combo !== 'escape') {
                return;
            }
        }

        // Execute handler
        try {
            shortcut.handler(event);

            // Visual feedback
            this._showFeedback(shortcut.description);
        } catch (error) {
            console.error('[Keyboard] Shortcut handler error:', error);
        }
    },

    /**
     * Show brief visual feedback for shortcut
     */
    _showFeedback(message) {
        // Create or get feedback element
        let feedback = document.getElementById('keyboard-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.id = 'keyboard-feedback';
            feedback.setAttribute('role', 'status');
            feedback.setAttribute('aria-live', 'polite');
            feedback.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                z-index: 10000;
                pointer-events: none;
                transition: opacity 0.2s ease;
            `;
            document.body.appendChild(feedback);
        }

        feedback.textContent = message;
        feedback.style.opacity = '1';

        // Fade out after delay
        clearTimeout(this._feedbackTimeout);
        this._feedbackTimeout = setTimeout(() => {
            feedback.style.opacity = '0';
        }, 1500);
    },

    /**
     * Get all shortcuts grouped by category
     */
    getShortcutsByCategory() {
        const categories = {};

        for (const shortcut of Object.values(this._shortcuts)) {
            const category = shortcut.category;
            if (!categories[category]) {
                categories[category] = [];
            }
            categories[category].push(shortcut);
        }

        return categories;
    },

    // --- Action handlers ---

    /**
     * Toggle recording start/stop
     */
    _toggleRecording() {
        if (typeof RecordingStateMachine !== 'undefined') {
            if (RecordingStateMachine.canPerformAction('stop')) {
                App.stopRecording();
            } else if (RecordingStateMachine.canPerformAction('start')) {
                App.startRecording();
            }
        } else {
            // Fallback without state machine
            if (App.state === 'recording') {
                App.stopRecording();
            } else if (App.state === 'idle' || App.state === 'done') {
                App.startRecording();
            }
        }
    },

    /**
     * Save current transcription
     */
    _saveTranscription() {
        // Check if there's a transcription to save
        if (typeof AutoSave !== 'undefined' && AutoSave.hasPendingChanges()) {
            AutoSave.saveNow();
        } else {
            this._showFeedback('No changes to save');
        }
    },

    /**
     * Refresh device list
     */
    _refreshDevices() {
        if (typeof App !== 'undefined' && App.loadDevices) {
            App.loadDevices();
        }
    },

    /**
     * Focus search input
     */
    _focusSearch() {
        const searchInput = document.querySelector('#search-input, [type="search"], .search-input');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        } else {
            this._showFeedback('No search field found');
        }
    },

    /**
     * Handle escape key
     */
    _handleEscape() {
        // Close any open modals
        const openModal = document.querySelector('.modal.is-active, .modal.open, [role="dialog"][aria-hidden="false"]');
        if (openModal) {
            const closeButton = openModal.querySelector('.close, [aria-label="Close"]');
            if (closeButton) {
                closeButton.click();
            }
            return;
        }

        // Clear focus from inputs
        if (document.activeElement && this._blockingElements.includes(document.activeElement.tagName)) {
            document.activeElement.blur();
            return;
        }

        // Close all notifications
        if (typeof UI !== 'undefined' && UI.dismissAllNotifications) {
            UI.dismissAllNotifications();
        }
    },

    /**
     * Reset app to idle
     */
    _resetApp() {
        if (typeof App !== 'undefined' && App.reset) {
            App.reset();
        }
    },

    /**
     * Show keyboard shortcuts help
     */
    _showHelp() {
        const categories = this.getShortcutsByCategory();

        let html = '<div class="keyboard-help">';
        html += '<h2>Keyboard Shortcuts</h2>';

        for (const [category, shortcuts] of Object.entries(categories)) {
            html += `<h3>${category}</h3>`;
            html += '<dl>';
            for (const shortcut of shortcuts) {
                const displayCombo = shortcut.combo
                    .split('+')
                    .map(key => `<kbd>${key}</kbd>`)
                    .join(' + ');
                html += `<dt>${displayCombo}</dt>`;
                html += `<dd>${shortcut.description}</dd>`;
            }
            html += '</dl>';
        }

        html += '</div>';

        // Create modal if UI supports it
        if (typeof UI !== 'undefined' && UI.showModal) {
            UI.showModal(html);
        } else {
            // Fallback: create simple modal
            this._createHelpModal(html);
        }
    },

    /**
     * Create simple help modal
     */
    _createHelpModal(content) {
        // Remove existing modal
        const existing = document.getElementById('keyboard-help-modal');
        if (existing) {
            existing.remove();
        }

        const modal = document.createElement('div');
        modal.id = 'keyboard-help-modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-label', 'Keyboard shortcuts');
        modal.innerHTML = `
            <div class="keyboard-help-backdrop"></div>
            <div class="keyboard-help-content">
                ${content}
                <button class="keyboard-help-close" aria-label="Close">×</button>
            </div>
        `;

        // Styles
        const style = document.createElement('style');
        style.textContent = `
            #keyboard-help-modal {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                z-index: 10001;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .keyboard-help-backdrop {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
            }
            .keyboard-help-content {
                position: relative;
                background: white;
                border-radius: 8px;
                padding: 24px;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            }
            .keyboard-help-content h2 {
                margin-top: 0;
                margin-bottom: 16px;
            }
            .keyboard-help-content h3 {
                margin: 16px 0 8px;
                font-size: 14px;
                text-transform: uppercase;
                color: #666;
            }
            .keyboard-help-content dl {
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 8px 16px;
                margin: 0;
            }
            .keyboard-help-content dt {
                font-weight: normal;
            }
            .keyboard-help-content dd {
                margin: 0;
                color: #333;
            }
            .keyboard-help-content kbd {
                display: inline-block;
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 12px;
                font-family: monospace;
            }
            .keyboard-help-close {
                position: absolute;
                top: 8px;
                right: 8px;
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                padding: 4px 8px;
                color: #666;
            }
            .keyboard-help-close:hover {
                color: #000;
            }
        `;
        document.head.appendChild(style);
        document.body.appendChild(modal);

        // Close handlers
        const close = () => {
            modal.remove();
            style.remove();
        };

        modal.querySelector('.keyboard-help-backdrop').addEventListener('click', close);
        modal.querySelector('.keyboard-help-close').addEventListener('click', close);

        // Close on escape (handled by main handler, but this is backup)
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                close();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    },
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KeyboardShortcuts.init());
} else {
    KeyboardShortcuts.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KeyboardShortcuts;
}
