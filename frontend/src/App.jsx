import React, { useState } from 'react';
import TopNav from './components/TopNav';
import Feed from './components/Feed';
import StateMonitor from './components/StateMonitor';

export default function App() {
  const [userId, setUserId] = useState(() => Math.floor(Math.random() * 90000) + 10000);
  const [history, setHistory] = useState([]);

  const handleResetUser = () => {
    setUserId(Math.floor(Math.random() * 90000) + 10000);
    setHistory([]);
  };

  const handleHistoryUpdate = (interaction) => {
    setHistory(prev => [interaction, ...prev]);
  };

  return (
    <div style={{ minHeight: '100vh' }}>
      <TopNav userId={userId} onResetUser={handleResetUser} />
      
      <div style={{ 
        display: 'flex', 
        gap: '2rem', 
        padding: '2rem', 
        maxWidth: '1200px', 
        margin: '0 auto',
        alignItems: 'flex-start'
      }}>
        <div style={{ flex: 1 }}>
          <Feed userId={userId} onHistoryUpdate={handleHistoryUpdate} />
        </div>
        
        <div style={{ position: 'sticky', top: '2rem' }}>
          <StateMonitor history={history} />
        </div>
      </div>
    </div>
  );
}