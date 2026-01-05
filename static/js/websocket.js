/**
 * WebSocket Client for CallWhisper
 * Handles real-time communication with the backend
 */

class WebSocketClient {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.handlers = new Map();
        this.isConnected = false;
        this._reconnectTimer = null;  // Track reconnect timer for proper cleanup
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            this.ws = new WebSocket(wsUrl);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Set up WebSocket event handlers
     */
    setupEventHandlers() {
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.emit('connection', { connected: true });
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this.isConnected = false;
            this.emit('connection', { connected: false });
            this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(data) {
        const type = data.type;

        // Emit specific event type
        if (type) {
            this.emit(type, data);
        }

        // Always emit 'message' for raw access
        this.emit('message', data);
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            return;
        }

        // Clear any existing reconnect timer
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            if (!this.isConnected) {
                this.connect();
            }
        }, delay);
    }

    /**
     * Send message to server
     */
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            if (typeof message === 'object') {
                this.ws.send(JSON.stringify(message));
            } else {
                this.ws.send(message);
            }
        }
    }

    /**
     * Send ping to keep connection alive
     */
    ping() {
        this.send('ping');
    }

    /**
     * Request current state
     */
    requestState() {
        this.send('state');
    }

    /**
     * Register event handler
     */
    on(event, handler) {
        if (!this.handlers.has(event)) {
            this.handlers.set(event, []);
        }
        this.handlers.get(event).push(handler);
    }

    /**
     * Remove event handler
     */
    off(event, handler) {
        if (this.handlers.has(event)) {
            const handlers = this.handlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    /**
     * Emit event to handlers
     */
    emit(event, data) {
        if (this.handlers.has(event)) {
            for (const handler of this.handlers.get(event)) {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in ${event} handler:`, error);
                }
            }
        }
    }

    /**
     * Close connection
     */
    close() {
        // Clear any pending reconnect timer
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
        }
    }
}

// Global WebSocket client instance
const wsClient = new WebSocketClient();
