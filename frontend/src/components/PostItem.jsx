import React, { useState, useEffect, useRef } from 'react';

export default function PostItem({ post, onInteract, isActive }) {
  const [watchTimeMs, setWatchTimeMs] = useState(0);
  const [isMuted, setIsMuted] = useState(true); 
  const [hasAudio, setHasAudio] = useState(true); 
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [canScroll, setCanScroll] = useState(false);
  
  const containerRef = useRef(null);
  const startTimeRef = useRef(null);
  const videoRef = useRef(null);
  const audioRef = useRef(null);
  const content = post.content;

  let audioUrl = null;
  if (content.videoUrl) {
    const baseUrl = content.videoUrl.split('?')[0]; 
    const urlParts = baseUrl.split('/');
    urlParts.pop(); 
    audioUrl = urlParts.join('/') + '/DASH_audio.mp4'; 
  }

  useEffect(() => {
    let interval;
    if (isActive) {
      startTimeRef.current = Date.now();
      interval = setInterval(() => {
        setWatchTimeMs(Date.now() - startTimeRef.current);
      }, 100); 
    }
    return () => clearInterval(interval);
  }, [isActive]);

  useEffect(() => {
    if (content.videoUrl && videoRef.current) {
      if (isActive) {
        videoRef.current.play().catch(err => console.warn("Video blocked:", err));
        if (audioRef.current && hasAudio) {
          audioRef.current.play().catch(err => {
            if (err.name === 'NotSupportedError') setHasAudio(false);
          });
        }
      } else {
        videoRef.current.pause();
        videoRef.current.currentTime = 0; 
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        }
      }
    }
  }, [isActive, content.videoUrl, hasAudio]);

  useEffect(() => {
    if (isActive) {
      const timer = setTimeout(() => setCanScroll(true), 500);
      return () => clearTimeout(timer);
    } else {
      setCanScroll(false);
    }
  }, [isActive]);

  useEffect(() => {
    const handleNativeWheel = (e) => {
      if (!isActive) return;
      e.preventDefault(); 
      
      if (!canScroll) return;
      if (e.deltaY > 40) {
        handleAction('SKIP');
      }
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener('wheel', handleNativeWheel, { passive: false });
    }
    return () => {
      if (container) container.removeEventListener('wheel', handleNativeWheel);
    };
  }, [isActive, canScroll]);

  let touchStartX = 0;
  let touchStartY = 0;
  const handleTouchStart = (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  };
  
  const handleTouchEnd = (e) => {
    if (!isActive || !canScroll) return;
    const touchEndX = e.changedTouches[0].clientX;
    const touchEndY = e.changedTouches[0].clientY;
    const diffX = Math.abs(touchStartX - touchEndX);
    const diffY = touchStartY - touchEndY; 
    
    if (diffY > 50 && diffY > diffX) {
      handleAction('SKIP');
    }
  };

  const calculateReward = (explicitAction = null) => {
    if (explicitAction === 'DISLIKE') return 0.0;

    const watchedSeconds = watchTimeMs / 1000;
    
    // Calculate Expected Duration
    let duration = 5.0;
    if (content.videoUrl && videoRef.current) {
      duration = videoRef.current.duration || 1.0; 
    } else if (content.imageUrls && content.imageUrls.length > 0) {
      duration = content.imageUrls.length * 3.0; // 3s per image
    } else {
      const fullText = (content.title || "") + " " + (content.text || "");
      const wordCount = fullText.trim().split(/\s+/).length;
      duration = Math.max(3.0, wordCount / 4.0); // 4 words per second, minimum 3s
    }

    // Convert Time to Equivalent Dataset Visits
    let equivalentVisits = (watchedSeconds / duration) * 2.0;
    equivalentVisits += (watchedSeconds / 5.0);

    if (explicitAction === 'LIKE') {
      equivalentVisits += 15.0; 
    }

    let baseReward = Math.log(1 + equivalentVisits);

    // Zero-Reward Grace Period
    if (watchedSeconds < 1.5 && explicitAction !== 'LIKE') {
      baseReward = 0.0;
    }
    
    // Envia apenas a reward base.
    // O backend aplica baseReward^2, penalização por repetição e bónus de diversidade.
    return parseFloat(baseReward.toFixed(4));
  };

  const handleAction = (actionType) => {
    const reward = calculateReward(actionType);
    onInteract(post, reward);
  };

  const toggleMute = (e) => {
    e.stopPropagation(); 
    setIsMuted(prev => !prev);
  };

  const handleNextImage = (e) => {
    e.preventDefault(); e.stopPropagation();
    setCurrentImageIndex((prev) => Math.min(prev + 1, content.imageUrls.length - 1));
  };
  
  const handlePrevImage = (e) => {
    e.preventDefault(); e.stopPropagation();
    setCurrentImageIndex((prev) => Math.max(prev - 1, 0));
  };

  const actionButtonStyle = {
    width: '50px', height: '50px', borderRadius: '50%',
    border: '1px solid rgba(255, 255, 255, 0.2)', backgroundColor: 'rgba(0, 0, 0, 0.4)',
    backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', color: 'white',
    cursor: 'pointer', fontSize: '24px', display: 'flex', alignItems: 'center',
    justifyContent: 'center', boxShadow: '0 4px 15px rgba(0, 0, 0, 0.3)'
  };

  return (
    <div 
      ref={containerRef}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      style={{
        position: 'absolute',
        left: 0, right: 0, margin: '0 auto',
        top: '10vh', 
        width: '100%', maxWidth: '500px', height: '80vh',
        backgroundColor: 'black', color: 'white', borderRadius: '12px',
        overflow: 'hidden', display: 'flex', flexDirection: 'column',
        touchAction: 'none', 
        opacity: isActive ? 1 : 0,
        visibility: isActive ? 'visible' : 'hidden',
        pointerEvents: isActive ? 'auto' : 'none',
        zIndex: isActive ? 10 : 0,
        transition: 'opacity 0.2s ease-in-out' 
    }}>
      
      <div style={{ padding: '15px', zIndex: 10, background: 'linear-gradient(rgba(0,0,0,0.8), transparent)' }}>
        <h3 style={{ margin: 0, color: '#ff4500' }}>r/{post.subreddit_name}</h3>
        <p style={{ margin: '5px 0 0 0', fontSize: '14px', fontWeight: 'bold', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{content.title}</p>
      </div>

      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
        {content.videoUrl ? (
          <>
            <video 
              ref={videoRef} src={content.videoUrl} loop muted playsInline 
              style={{ width: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
            {hasAudio && (
              <audio ref={audioRef} src={audioUrl} loop muted={isMuted} onError={() => setHasAudio(false)} />
            )}
            {hasAudio && (
              <button onClick={toggleMute} style={{ ...actionButtonStyle, position: 'absolute', top: '15px', right: '15px', zIndex: 20 }}>
                {isMuted ? '🔇' : '🔊'}
              </button>
            )}
          </>
        ) : content.imageUrls && content.imageUrls.length > 0 ? (
          <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
            <div style={{ 
              display: 'flex', width: '100%', height: '100%',
              transform: `translateX(-${currentImageIndex * 100}%)`,
              transition: 'transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)' 
            }}>
              {content.imageUrls.map((url, idx) => (
                <img key={idx} src={url} alt={`slide-${idx}`} style={{ width: '100%', height: '100%', flexShrink: 0, objectFit: 'contain' }} />
              ))}
            </div>
            {content.imageUrls.length > 1 && (
              <>
                {currentImageIndex > 0 && <button onClick={handlePrevImage} style={{ ...actionButtonStyle, position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', zIndex: 20 }}>◀</button>}
                {currentImageIndex < content.imageUrls.length - 1 && <button onClick={handleNextImage} style={{ ...actionButtonStyle, position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', zIndex: 20 }}>▶</button>}
                <div style={{ position: 'absolute', bottom: '15px', width: '100%', display: 'flex', justifyContent: 'center', gap: '8px' }}>
                  {content.imageUrls.map((_, idx) => (
                    <div key={idx} style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: idx === currentImageIndex ? 'white' : 'rgba(255,255,255,0.4)', transition: 'background-color 0.3s' }} />
                  ))}
                </div>
              </>
            )}
          </div>
        ) : (
          <div style={{ padding: '30px', textAlign: 'left', color: '#ccc', maxHeight: '100%', overflowY: 'auto', lineHeight: '1.6', fontSize: '15px' }}>
            {content.text ? content.text : 'Link / Text Post'}
          </div>
        )}
      </div>

      {isActive && (
        <div style={{
          position: 'absolute', right: '15px', bottom: '15%', display: 'flex', flexDirection: 'column', gap: '15px', zIndex: 30
        }}>
          <button onClick={() => handleAction('LIKE')} style={actionButtonStyle} title="Like">👍</button>
          <button onClick={() => handleAction('DISLIKE')} style={actionButtonStyle} title="Dislike">👎</button>
        </div>
      )}

      {isActive && (
        <div style={{ position: 'absolute', bottom: 10, left: 10, fontSize: '12px', color: '#00ff00', fontFamily: 'monospace', zIndex: 30, background: 'rgba(0,0,0,0.5)', padding: '2px 6px', borderRadius: '4px' }}>
          Live Reward Calc: {calculateReward()}
        </div>
      )}
    </div>
  );
}