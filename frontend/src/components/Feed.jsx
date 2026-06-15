import React, { useState, useEffect, useRef } from 'react';
import PostItem from './PostItem';
import { fetchNextRecommendation, sendInteraction } from '../services/api';

const PRELOAD_COUNT = 3;

export default function Feed({ userId, onHistoryUpdate }) {
  const [queue, setQueue] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const seenPostIds = useRef(new Set());

const fetchRealRedditPost = async (subredditName) => {
    try {
      const res = await fetch(`http://localhost:8000/api/reddit/${subredditName}`);
      
      if (!res.ok) throw new Error("Network response was not OK");
      
      const json = await res.json();
      const posts = json.data.children;
      
      for (let p of posts) {
        const postData = p.data;
        if (postData.stickied || seenPostIds.current.has(postData.id)) continue;
        
        seenPostIds.current.add(postData.id);
        
        const videoUrl = postData.secure_media?.reddit_video?.fallback_url || null;
        
        return {
          redditId: postData.id,
          title: postData.title,
          author: postData.author,
          videoUrl: videoUrl,
          imageUrls: postData.image_urls || [],
          text: postData.selftext
        };
      }
    } catch (e) {
      console.error(`Failed to fetch from r/${subredditName}`, e);
    }
    
    return {
      redditId: `fallback-${Date.now()}`,
      title: `No media found for r/${subredditName}`,
      author: "System",
      videoUrl: null,
      imageUrl: null,
      text: "Could not load content for this recommendation."
    };
  };

  useEffect(() => {
    let isMounted = true;
    
    const fillBuffer = async () => {
      setLoading(true);
      setQueue([]); 
      seenPostIds.current.clear();
      try {
        const newItems = [];
        for (let i = 0; i < PRELOAD_COUNT; i++) {
          const rec = await fetchNextRecommendation(userId);
          const realContent = await fetchRealRedditPost(rec.subreddit_name);
          
          newItems.push({ 
            ...rec, 
            content: realContent,
            queueId: crypto.randomUUID() 
          });
        }
        if (isMounted) setQueue(newItems);
      } catch (error) {
        console.error(error);
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fillBuffer();
    return () => { isMounted = false; };
  }, [userId]);

  const handleInteract = async (post, baseReward) => {
    setQueue(prev => prev.slice(1));
  
    try {
      const interactionResult = await sendInteraction(userId, post.sub_id, baseReward);
  
      const shapedReward = interactionResult.shaped_reward ?? baseReward;
  
      onHistoryUpdate({
        sub_id: post.sub_id,
        subreddit_name: post.subreddit_name,
        reward: shapedReward,
        base_reward: baseReward,
        repeated: interactionResult.repeated
      });
  
      const newRec = await fetchNextRecommendation(userId);
      const realContent = await fetchRealRedditPost(newRec.subreddit_name);
  
      setQueue(prev => [
        ...prev,
        { ...newRec, content: realContent, queueId: crypto.randomUUID() }
      ]);
    } catch (error) {
      console.error("Failed to process RL interaction loop:", error);
    }
  };

  if (loading) return <div>Initializing feed...</div>;
  if (queue.length === 0) return <div>No recommendations available.</div>;

return (
    <div style={{ position: 'relative', maxWidth: '500px', margin: '0 auto', height: '100vh', overflow: 'hidden' }}>
      {queue.map((post, index) => (
        <PostItem 
          key={post.queueId} 
          post={post} 
          isActive={index === 0} 
          onInteract={handleInteract} 
        />
      ))}
    </div>
  );
}