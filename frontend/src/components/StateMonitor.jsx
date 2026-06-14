import React from 'react';

export default function StateMonitor({ history }) {
  return (
    <div style={{
      width: '300px',
      backgroundColor: '#1c1c1c',
      color: '#00ff00',
      fontFamily: 'monospace',
      padding: '1rem',
      borderRadius: '8px',
      height: 'calc(100vh - 120px)',
      overflowY: 'auto'
    }}>
      <h3 style={{ marginTop: 0, color: 'white', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>
        Agent Memory State
      </h3>
      
      {history.length === 0 ? (
        <div style={{ color: '#666' }}>[Array Empty - Waiting for actions]</div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {history.map((item, index) => (
            <li key={index} style={{ marginBottom: '1rem', padding: '0.5rem', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
              <div style={{ color: 'white' }}>Action {history.length - index}</div>
              <div>Sub: <span style={{ color: '#ffcc00' }}>{item.subreddit_name}</span> ({item.sub_id})</div>
              <div>Reward: <span style={{ color: item.reward > 0 ? '#4CAF50' : '#f44336' }}>{item.reward.toFixed(1)}</span></div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}