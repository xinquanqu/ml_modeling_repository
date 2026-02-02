import { useState, useEffect, useCallback, useRef } from 'react';

export function useWebSocket(url, onMessage) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const reconnectTimeoutRef = useRef(null);

  // Update ref when callback changes
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const connect = useCallback(() => {
    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Don't connect if already connected/connecting
    if (wsRef.current) {
      if (wsRef.current.readyState < 2) {
        console.log("WebSocket already open/connecting");
        return;
      }
      wsRef.current.close();
    }

    console.log(`ws: Connecting to ${url}`);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("ws: Connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessageRef.current) {
          onMessageRef.current(data);
        }
      } catch (err) {
        console.error("ws: Parse error", err);
      }
    };

    ws.onclose = (event) => {
      console.log("ws: Disconnected", event.code, event.reason);
      setIsConnected(false);
      wsRef.current = null;

      // Auto-reconnect
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log("ws: Attempting reconnect...");
        connect();
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error("ws: Error", error);
      ws.close();
    };

  }, [url]);

  // Initial connect
  useEffect(() => {
    connect();

    return () => {
      console.log("ws: Cleanup");
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        // Remove listeners to prevent reconnect loop during unmount
        // But for strict mode, we might WANT them? 
        // Best practice: cleanup means WE are done.
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
    };
  }, [connect]);

  const sendMessage = useCallback((msg) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log("ws: Sending", msg);
      wsRef.current.send(JSON.stringify(msg));
    } else {
      console.warn("ws: Not connected! State:", wsRef.current ? wsRef.current.readyState : "NULL", msg);
      // Optional: Queue message? For now just log.
    }
  }, []);

  return { isConnected, sendMessage, connect };
}
