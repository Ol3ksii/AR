import React from 'react';

export default function TopNav({ userId, onResetUser }) {
  return (
    <header style={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center', 
      padding: '1rem 2rem', 
      backgroundColor: '#ff4500', 
      color: 'white' 
    }}>
      <h1 style={{ margin: 0, fontSize: '1.5rem' }}>RL Recommender Feed</h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <span style={{ fontWeight: 'bold' }}>Current User: {userId}</span>
        <button 
          onClick={onResetUser}
          style={{
            padding: '8px 16px',
            backgroundColor: 'white',
            color: '#ff4500',
            border: 'none',
            borderRadius: '4px',
            fontWeight: 'bold',
            cursor: 'pointer'
          }}
        >
          Simulate Brand New User
        </button>
      </div>
    </header>
  );
}