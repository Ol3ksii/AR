const API_BASE_URL = 'http://localhost:8000/api';
// -------------------------------------

export const fetchNextRecommendation = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/recommend/${userId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch recommendation');
  }
  return await response.json();
};

export const sendInteraction = async (userId, subId, reward) => {
  const response = await fetch(`${API_BASE_URL}/interact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      sub_id: subId,
      reward: reward,
    }),
  });
  if (!response.ok) {
    throw new Error('Failed to post interaction');
  }
  return await response.json();
};