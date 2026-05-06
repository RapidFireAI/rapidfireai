/**
 * Stub for WebhooksSettings — not available in the RapidFire fork.
 * Renders nothing so gateway/BudgetsPage compiles without changes.
 */
import React from 'react';

interface WebhooksSettingsProps {
  eventFilter?: string;
  title?: React.ReactNode;
  description?: React.ReactNode;
  componentIds?: Record<string, string>;
}

const WebhooksSettings: React.FC<WebhooksSettingsProps> = () => null;

export default WebhooksSettings;
