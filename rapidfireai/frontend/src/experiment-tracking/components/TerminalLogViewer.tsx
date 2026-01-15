import React, { useState, useEffect, useRef } from 'react';

interface TerminalLogViewerProps {
    logs: string[];
    onScrollToBottom?: () => void;
    emptyStateMessage?: string;
}

const TerminalLogViewer: React.FC<TerminalLogViewerProps> = ({ 
    logs, 
    onScrollToBottom,
    emptyStateMessage = "No logs available. If this is a new cluster, logs will appear here once the cluster starts."
}) => {
    const [logLines, setLogLines] = useState<string[]>([]);
    const logContainerRef = useRef<HTMLDivElement | null>(null);
    const [autoScroll, setAutoScroll] = useState(true);

    useEffect(() => {
        setLogLines(logs);
    }, [logs]);

    useEffect(() => {
        if (logContainerRef.current && autoScroll) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logLines, autoScroll]);

    const handleScroll = () => {
        if (!logContainerRef.current) return;

        const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 50;
        
        // Update auto-scroll based on whether we're near the bottom
        setAutoScroll(isNearBottom);

        // If we're near the bottom and have a scroll handler, call it
        if (isNearBottom && onScrollToBottom) {
            onScrollToBottom();
        }
    };

    return (
        <div
            ref={logContainerRef}
            className="terminal-container"
            style={{
                backgroundColor: '#000',
                color: '#fff',
                fontFamily: 'monospace',
                padding: '20px',
                height: '100%',
                width: '100%',
                overflowY: 'auto',
                borderRadius: '5px',
                boxShadow: '0 0 10px rgba(0,0,0,0.5)',
                position: 'relative',
            }}
            onScroll={handleScroll}
        >
            {logLines.length > 0 ? (
                <div style={{ minHeight: '100%' }}>
                    {logLines.map((line, index) => (
                        <pre key={index} style={{ 
                            margin: 0, 
                            whiteSpace: 'pre-wrap', 
                            wordBreak: 'break-all',
                            lineHeight: '1.5'
                        }}>
                            {line}
                        </pre>
                    ))}
                </div>
            ) : (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center',
                    color: '#888',
                    width: '80%'
                }}>
                    {emptyStateMessage}
                </div>
            )}
        </div>
    );
};

export default TerminalLogViewer;
