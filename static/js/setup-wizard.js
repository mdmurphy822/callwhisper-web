/**
 * CallWhisper First-Run Setup Wizard
 *
 * Guides users through VB-Cable installation on first launch.
 * Checks for virtual audio devices and prompts installation if needed.
 *
 * Based on LibV2 accessibility patterns:
 * - Modal dialogs with focus trap
 * - ARIA live region announcements
 * - Keyboard navigation (Escape to close)
 *
 * Usage:
 *   SetupWizard.checkPrerequisites();  // Call on app init
 */

const SetupWizard = {
    // Configuration
    config: {
        apiEndpoint: '/api/setup/status',
        vbCableUrl: 'https://vb-audio.com/Cable/',
        localStorageKey: 'callwhisper_setup_complete',
    },

    // State
    _modalElement: null,
    _previousFocus: null,
    _recordingBlocked: true,  // Block recording until virtual audio resolved
    _handleEscapeKeyBound: null,  // Bound escape key handler for proper add/remove

    /**
     * Check prerequisites and show wizard if needed
     */
    async checkPrerequisites() {
        // Skip if already completed (localStorage check for quick path)
        if (this._isLocallyComplete()) {
            console.log('[Setup] Previously completed, skipping check');
            this._recordingBlocked = false;
            return true;
        }

        try {
            const response = await fetch(this.config.apiEndpoint);
            if (!response.ok) {
                console.warn('[Setup] Status endpoint not available');
                this._recordingBlocked = false;  // Don't block if endpoint fails
                return true;
            }

            const status = await response.json();
            console.log('[Setup] Status:', status);

            if (status.setup_complete) {
                this._markLocallyComplete();
                this._recordingBlocked = false;
                return true;
            }

            if (status.virtual_audio_detected) {
                // Virtual audio detected - show green status and enable recording
                this._showDetectedStatus(status);
                this._recordingBlocked = false;
                await this.markComplete(false);
                return true;
            }

            // No virtual audio - show blocking wizard
            this._recordingBlocked = true;
            this.showVirtualAudioPrompt(status);
            return false;

        } catch (error) {
            console.error('[Setup] Check error:', error);
            this._recordingBlocked = false;  // Don't block on error
            return true;
        }
    },

    /**
     * Show green status indicator when virtual audio is detected
     */
    _showDetectedStatus(status) {
        const deviceName = status.detected_devices?.[0]?.name || 'Virtual Audio Device';

        // Show success notification
        if (typeof UI !== 'undefined' && UI.showSuccess) {
            UI.showSuccess(`Virtual audio detected: ${deviceName}`);
        }

        // Announce to screen readers
        this._announce(`Virtual audio device detected: ${deviceName}. Recording is now available.`);

        console.log('[Setup] Virtual audio detected:', deviceName);
    },

    /**
     * Show virtual audio setup prompt
     */
    showVirtualAudioPrompt(status) {
        // Save current focus for restoration
        this._previousFocus = document.activeElement;

        // Create modal HTML with 3-button layout
        const modalHtml = `
            <div id="setup-wizard-overlay" class="setup-overlay">
                <div id="setup-wizard-modal"
                     class="setup-modal"
                     role="dialog"
                     aria-modal="true"
                     aria-labelledby="setup-title"
                     aria-describedby="setup-description">

                    <h2 id="setup-title" class="setup-title">Virtual Audio Device Required</h2>

                    <div id="setup-description" class="setup-description">
                        <p>
                            To capture Jabber/Finesse call audio, a virtual audio device is required.
                        </p>

                        ${this._renderDeviceStatus(status)}

                        <div class="setup-recommendation">
                            <h3>Choose an Audio Capture Method</h3>
                            <p>
                                You have two options to capture call audio:
                            </p>
                        </div>
                    </div>

                    <div class="setup-actions setup-actions-vertical">
                        <button id="setup-install-btn"
                                class="setup-btn setup-btn-primary"
                                onclick="SetupWizard.openVBCable()">
                            Install VB-Cable (Recommended)
                        </button>
                        <button id="setup-stereo-btn"
                                class="setup-btn setup-btn-secondary"
                                onclick="SetupWizard.showStereoMixInstructions()">
                            Enable Windows Stereo Mix
                        </button>
                        <button id="setup-instructions-btn"
                                class="setup-btn setup-btn-tertiary"
                                onclick="SetupWizard.showInstructions()">
                            View Detailed Instructions
                        </button>
                        <button id="setup-continue-btn"
                                class="setup-btn setup-btn-tertiary"
                                onclick="SetupWizard.continueWithoutRecording()">
                            Continue Without Recording
                        </button>
                    </div>

                    <div class="setup-note">
                        <p>
                            <strong>Note:</strong> VB-Cable requires administrator rights to install.
                            Contact your IT department if you cannot install software.
                        </p>
                        <p class="setup-enterprise-note">
                            <em>For managed environments, IT may deploy VB-Cable separately via SCCM/Intune.
                            CallWhisper will auto-detect the device once present.</em>
                        </p>
                    </div>

                    <button class="setup-close"
                            type="button"
                            aria-label="Close setup wizard"
                            onclick="SetupWizard.continueWithoutRecording()">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
            </div>
        `;

        // Insert modal
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        this._modalElement = document.getElementById('setup-wizard-overlay');

        // Add escape key handler (store bound reference for proper removal)
        this._handleEscapeKeyBound = this._handleEscapeKey.bind(this);
        document.addEventListener('keydown', this._handleEscapeKeyBound);

        // Focus first action button
        const installBtn = document.getElementById('setup-install-btn');
        if (installBtn) {
            installBtn.focus();
        }

        // Trap focus in modal
        this._setupFocusTrap();

        // Announce to screen readers
        this._announce('Virtual audio device setup required. Recording is disabled until resolved. Press Tab to navigate options.');
    },

    /**
     * Render device status section
     */
    _renderDeviceStatus(status) {
        const devices = status.all_audio_devices || [];
        const virtualDevices = status.detected_devices || [];

        if (virtualDevices.length > 0) {
            // Has some virtual device but maybe not recommended
            return `
                <div class="setup-status setup-status-warning">
                    <span class="setup-status-icon" aria-hidden="true">⚠️</span>
                    <span>
                        Found: ${virtualDevices.map(d => d.name).join(', ')}<br>
                        <em>Consider upgrading to VB-Cable for better reliability.</em>
                    </span>
                </div>
            `;
        }

        return `
            <div class="setup-status setup-status-error">
                <span class="setup-status-icon" aria-hidden="true">❌</span>
                <span>No virtual audio device detected</span>
            </div>
            <p class="setup-detail">
                ${devices.length} audio devices found, but none are virtual audio cables.
            </p>
        `;
    },

    /**
     * Open VB-Cable download page
     */
    openVBCable() {
        // Open in new tab
        window.open(this.config.vbCableUrl, '_blank', 'noopener,noreferrer');

        // Show post-install instructions
        this.showPostInstallInstructions();
    },

    /**
     * Show post-installation instructions (after clicking Install VB-Cable)
     */
    showPostInstallInstructions() {
        const modal = document.getElementById('setup-wizard-modal');
        if (!modal) return;

        modal.innerHTML = `
            <h2 id="setup-title" class="setup-title">After Installing VB-Cable</h2>

            <div id="setup-description" class="setup-description">
                <ol class="setup-steps">
                    <li>Download VB-Cable from the opened page</li>
                    <li>Run the installer (requires administrator rights)</li>
                    <li>Restart your computer when prompted</li>
                    <li>Re-launch CallWhisper</li>
                </ol>

                <div class="setup-recommendation">
                    <h3>Configure Your Audio</h3>
                    <p>
                        After VB-Cable is installed, set your softphone (Jabber/Finesse)
                        to output audio to "CABLE Input". CallWhisper will automatically
                        detect "CABLE Output" for recording.
                    </p>
                </div>
            </div>

            <div class="setup-actions">
                <button id="setup-recheck-btn"
                        class="setup-btn setup-btn-primary"
                        onclick="SetupWizard.recheckDevices()">
                    Check Again
                </button>
                <button id="setup-continue-btn"
                        class="setup-btn setup-btn-tertiary"
                        onclick="SetupWizard.continueWithoutRecording()">
                    Continue Without Recording
                </button>
            </div>

            <button class="setup-close"
                    type="button"
                    aria-label="Close setup wizard"
                    onclick="SetupWizard.continueWithoutRecording()">
                <span aria-hidden="true">&times;</span>
            </button>
        `;

        // Focus recheck button
        const recheckBtn = document.getElementById('setup-recheck-btn');
        if (recheckBtn) {
            recheckBtn.focus();
        }
    },

    /**
     * Show Stereo Mix setup instructions
     */
    showStereoMixInstructions() {
        const modal = document.getElementById('setup-wizard-modal');
        if (!modal) return;

        modal.innerHTML = `
            <h2 id="setup-title" class="setup-title">Enable Windows Stereo Mix</h2>

            <div id="setup-description" class="setup-description setup-instructions-detail">
                <div class="setup-status setup-status-warning">
                    <span class="setup-status-icon" aria-hidden="true">&#x2139;</span>
                    <span>Stereo Mix is a built-in Windows feature, but may not be available on all systems.</span>
                </div>

                <h3>Step 1: Open Sound Settings</h3>
                <ol class="setup-steps">
                    <li>Right-click the <strong>speaker icon</strong> in your taskbar</li>
                    <li>Select <strong>"Sounds"</strong> or <strong>"Sound settings"</strong></li>
                    <li>Click <strong>"More sound settings"</strong> if on Windows 11</li>
                </ol>

                <h3>Step 2: Enable Stereo Mix</h3>
                <ol class="setup-steps">
                    <li>Go to the <strong>"Recording"</strong> tab</li>
                    <li>Right-click in an empty area of the device list</li>
                    <li>Check <strong>"Show Disabled Devices"</strong></li>
                    <li>If "Stereo Mix" appears, right-click it and select <strong>"Enable"</strong></li>
                    <li>Right-click again and select <strong>"Set as Default Device"</strong> (optional)</li>
                </ol>

                <h3>Step 3: Use in CallWhisper</h3>
                <ol class="setup-steps">
                    <li>Click "Check Again" below to detect Stereo Mix</li>
                    <li>Select "Stereo Mix" as your recording device</li>
                    <li>Start recording before your call</li>
                </ol>

                <div class="setup-recommendation">
                    <h3>Stereo Mix Not Available?</h3>
                    <p>
                        Some sound card drivers don't include Stereo Mix. In this case,
                        use VB-Cable or VoiceMeeter as alternatives.
                    </p>
                </div>
            </div>

            <div class="setup-actions">
                <button id="setup-back-btn"
                        class="setup-btn setup-btn-secondary"
                        onclick="SetupWizard.goBackToMain()">
                    Back
                </button>
                <button id="setup-recheck-btn"
                        class="setup-btn setup-btn-primary"
                        onclick="SetupWizard.recheckDevices()">
                    Check for Stereo Mix
                </button>
            </div>

            <button class="setup-close"
                    type="button"
                    aria-label="Close setup wizard"
                    onclick="SetupWizard.continueWithoutRecording()">
                <span aria-hidden="true">&times;</span>
            </button>
        `;

        // Focus back button
        const backBtn = document.getElementById('setup-back-btn');
        if (backBtn) {
            backBtn.focus();
        }

        // Announce to screen readers
        this._announce('Showing Windows Stereo Mix setup instructions.');
    },

    /**
     * Show detailed setup instructions modal
     */
    showInstructions() {
        const modal = document.getElementById('setup-wizard-modal');
        if (!modal) return;

        modal.innerHTML = `
            <h2 id="setup-title" class="setup-title">VB-Cable Setup Instructions</h2>

            <div id="setup-description" class="setup-description setup-instructions-detail">
                <h3>Step 1: Download VB-Cable</h3>
                <ol class="setup-steps">
                    <li>Visit <a href="https://vb-audio.com/Cable/" target="_blank" rel="noopener">vb-audio.com/Cable</a></li>
                    <li>Download the VB-Cable Driver Package (VBCABLE_Driver_Pack*.zip)</li>
                    <li>Extract the ZIP file to a folder</li>
                </ol>

                <h3>Step 2: Install VB-Cable</h3>
                <ol class="setup-steps">
                    <li>Right-click <strong>VBCABLE_Setup_x64.exe</strong></li>
                    <li>Select "Run as administrator"</li>
                    <li>Click "Install Driver" when prompted</li>
                    <li>Wait for installation to complete</li>
                    <li><strong>Restart your computer</strong></li>
                </ol>

                <h3>Step 3: Configure Jabber/Finesse</h3>
                <ol class="setup-steps">
                    <li>Open Jabber or Finesse audio settings</li>
                    <li>Set <strong>Speaker</strong> to "CABLE Input (VB-Audio Virtual Cable)"</li>
                    <li>Keep <strong>Microphone</strong> as your headset (unchanged)</li>
                    <li>Test a call - caller audio goes to VB-Cable, your voice stays on headset</li>
                </ol>

                <h3>Step 4: Use CallWhisper</h3>
                <ol class="setup-steps">
                    <li>Launch CallWhisper</li>
                    <li>Select "CABLE Output" as the recording device</li>
                    <li>Click "Start Recording" before your call</li>
                    <li>Click "Stop + Transcribe" after your call</li>
                </ol>
            </div>

            <div class="setup-actions">
                <button id="setup-back-btn"
                        class="setup-btn setup-btn-secondary"
                        onclick="SetupWizard.goBackToMain()">
                    Back
                </button>
                <button id="setup-recheck-btn"
                        class="setup-btn setup-btn-primary"
                        onclick="SetupWizard.recheckDevices()">
                    Check for VB-Cable
                </button>
            </div>

            <button class="setup-close"
                    type="button"
                    aria-label="Close setup wizard"
                    onclick="SetupWizard.continueWithoutRecording()">
                <span aria-hidden="true">&times;</span>
            </button>
        `;

        // Focus back button
        const backBtn = document.getElementById('setup-back-btn');
        if (backBtn) {
            backBtn.focus();
        }

        // Announce to screen readers
        this._announce('Showing detailed VB-Cable setup instructions.');
    },

    /**
     * Go back to main setup prompt
     */
    async goBackToMain() {
        try {
            const response = await fetch(this.config.apiEndpoint);
            const status = await response.json();

            // Close current modal and show main prompt again
            this.closeModal();
            this.showVirtualAudioPrompt(status);
        } catch (error) {
            console.error('[Setup] Error going back:', error);
            // Just close and re-show with empty status
            this.closeModal();
            this.showVirtualAudioPrompt({ detected_devices: [], all_audio_devices: [] });
        }
    },

    /**
     * Re-check for audio devices after user installs VB-Cable
     */
    async recheckDevices() {
        const recheckBtn = document.getElementById('setup-recheck-btn');
        if (recheckBtn) {
            recheckBtn.disabled = true;
            recheckBtn.textContent = 'Checking...';
        }

        try {
            const response = await fetch(this.config.apiEndpoint);
            const status = await response.json();

            if (status.virtual_audio_detected) {
                // Success - unblock recording!
                this._recordingBlocked = false;
                await this.markComplete(false);
                this.closeModal();

                // Show success notification
                if (typeof UI !== 'undefined' && UI.showSuccess) {
                    UI.showSuccess('Virtual audio device detected! Recording is now available.');
                }

                // Refresh devices list and update button states
                if (typeof App !== 'undefined') {
                    if (App.refreshDevices) App.refreshDevices();
                    if (App.updateButtonStates) App.updateButtonStates();
                }

                // Announce to screen readers
                this._announce('Virtual audio detected. Recording is now enabled.');
            } else {
                // Still not detected
                if (typeof UI !== 'undefined' && UI.showWarning) {
                    UI.showWarning('Virtual audio not detected yet. Make sure VB-Cable is installed and restart if needed.');
                }

                if (recheckBtn) {
                    recheckBtn.disabled = false;
                    recheckBtn.textContent = 'Check Again';
                }
            }

        } catch (error) {
            console.error('[Setup] Recheck error:', error);
            if (recheckBtn) {
                recheckBtn.disabled = false;
                recheckBtn.textContent = 'Check Again';
            }
        }
    },

    /**
     * Continue without recording (recording stays blocked)
     * User explicitly chooses to proceed without virtual audio
     */
    async continueWithoutRecording() {
        // Keep recording blocked
        this._recordingBlocked = true;
        await this.markComplete(true);
        this.closeModal();

        // Show persistent warning
        if (typeof UI !== 'undefined' && UI.showWarning) {
            UI.showWarning('Recording disabled. Install VB-Cable to enable call recording.');
        }

        // Update button states to reflect blocked recording
        if (typeof App !== 'undefined' && App.updateButtonStates) {
            App.updateButtonStates();
        }

        // Announce to screen readers
        this._announce('Setup skipped. Recording is disabled until virtual audio is installed.');

        console.log('[Setup] User chose to continue without recording');
    },

    /**
     * Skip setup (legacy - redirects to continueWithoutRecording)
     * @deprecated Use continueWithoutRecording() instead
     */
    async skipSetup() {
        await this.continueWithoutRecording();
    },

    /**
     * Mark setup as complete
     */
    async markComplete(skipped) {
        this._markLocallyComplete();

        try {
            await fetch('/api/setup/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ skipped }),
            });
        } catch (error) {
            console.warn('[Setup] Could not save setup status to server:', error);
        }

        console.log('[Setup] Marked complete, skipped:', skipped);
    },

    /**
     * Close the modal
     */
    closeModal() {
        // Remove escape key handler
        if (this._handleEscapeKeyBound) {
            document.removeEventListener('keydown', this._handleEscapeKeyBound);
            this._handleEscapeKeyBound = null;
        }

        // Remove modal
        if (this._modalElement) {
            this._modalElement.remove();
            this._modalElement = null;
        }

        // Restore focus
        if (this._previousFocus) {
            this._previousFocus.focus();
            this._previousFocus = null;
        }
    },

    /**
     * Check localStorage for quick completion check
     */
    _isLocallyComplete() {
        return localStorage.getItem(this.config.localStorageKey) === 'true';
    },

    /**
     * Mark locally complete in localStorage
     */
    _markLocallyComplete() {
        localStorage.setItem(this.config.localStorageKey, 'true');
    },

    /**
     * Handle escape key to close modal
     */
    _handleEscapeKey(event) {
        if (event.key === 'Escape' && SetupWizard._modalElement) {
            SetupWizard.skipSetup();
        }
    },

    /**
     * Set up focus trap within modal
     */
    _setupFocusTrap() {
        const modal = document.getElementById('setup-wizard-modal');
        if (!modal) return;

        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );

        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        modal.addEventListener('keydown', (event) => {
            if (event.key !== 'Tab') return;

            if (event.shiftKey) {
                // Shift + Tab
                if (document.activeElement === firstElement) {
                    event.preventDefault();
                    lastElement.focus();
                }
            } else {
                // Tab
                if (document.activeElement === lastElement) {
                    event.preventDefault();
                    firstElement.focus();
                }
            }
        });
    },

    /**
     * Announce message to screen readers
     */
    _announce(message) {
        const announcer = document.getElementById('status-announcer');
        if (announcer) {
            announcer.textContent = '';
            setTimeout(() => {
                announcer.textContent = message;
            }, 100);
        }
    },

    /**
     * Reset setup (for testing/debugging)
     */
    reset() {
        localStorage.removeItem(this.config.localStorageKey);
        console.log('[Setup] Reset complete');
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SetupWizard;
}
