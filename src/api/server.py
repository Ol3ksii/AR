import os
import json
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
import uvicorn
import sys
import time
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.dqn import DQNAgent
from src.config import HISTORY_LENGTH

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

ml_models = {}
id2sub = {}
user_sessions = {}
reddit_cache = {}

# --- MANUAL CONFIGURATION REQUIRED ---
REDDIT_BROWSER_COOKIE = """csv=2; edgebucket=gh754Blq16Q8iRXlgf; __stripe_mid=5ec03638-f2b9-4c7e-99d4-92f56b927d941b787a; reddit_session=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpsVFdYNlFVUEloWktaRG1rR0pVd1gvdWNFK01BSjBYRE12RU1kNzVxTXQ4IiwidHlwIjoiSldUIn0.eyJzdWIiOiJ0Ml8yZ2Y1OWo4NTZlIiwiZXhwIjoxNzk3MDAxNjM2LjU3NDcwNSwiaWF0IjoxNzgxMzYzMjM2LjU3NDcwNSwianRpIjoiZ3hZTldiUjdxUE5qS1hkRGdob0E0UWI4S1FUTGVRIiwiYXQiOjEsImNpZCI6ImNvb2tpZSIsImxjYSI6MTc4MTM2MzIzNTgxMywic2NwIjoiZUp5S2pnVUVBQURfX3dFVkFMayIsImZsbyI6NywiYW1yIjpbInNzbyJdfQ.Fx--kOX6ipZig9YEOJhIDdd5TBxf2QVpg5a8OSfBd99NEj3khiIvR340b53-Xx6m8YjgMZjyLcVb4OC6UCxiatmh_uPg8mVTq1-DGX6ox0HgP__r7cgkOiGp0gLA6oDbRxRHve80u7WW8EPnq0b42IDiAzuqpZcqQyXw0cH_xgIx7MWLB_HspsJ2GllFmmcXIBJwshjtywo6y4Tg60GbocHe2ZK1BZ8vkerp5FtIkPzgn3PpuxccsfUzB_Oat4HJXdAPtoEuAznSaQAoFLQueclCiINTNFjW49Dswvsy3qicKHfu6W9SN5GkCjKKeRzoAY1vT5tXvb8UEcuAKP77jA; loid=000000002gf59j856e.2.1781363235813.Z0FBQUFBQnFMWElrazNKTEJ4Z2wwOURWaWVma1Y5WkdkRzdjMERsTXZUTjlGX1ppbkFmUGt5WS03S3F4UXFiWW12a2ZjMDc2d2h2NWt0ZFZvRFdxNzZtT0hfOEFhcVFfN215c1N2NXR2ajBVSWRGdnR4LTlqT3J1OW1vZ1hZSHBGaHkySnFrU2tmRG8; devvit_games_drawer_expanded_state=0; pc=qs; eu_cookie={%22opted%22:true%2C%22nonessential%22:true}; _ga=GA1.1.316803955.1781363660; reddit_chat_view=closed; token_v2=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpzS3dsMnlsV0VtMjVmcXhwTU40cWY4MXE2OWFFdWFyMnpLMUdhVGxjdWNZIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNzgxNDUwMTE2LjY5MTY2NSwiaWF0IjoxNzgxMzYzNzE2LjY5MTY2NSwianRpIjoiRkJYU1gwWWF3dFRJUC1nckx6U2dWZUdxNkVSMDVBIiwiY2lkIjoiMFItV0FNaHVvby1NeVEiLCJsaWQiOiJ0Ml8yZ2Y1OWo4NTZlIiwiYWlkIjoidDJfMmdmNTlqODU2ZSIsImF0IjoxLCJsY2EiOjE3ODEzNjMyMzU4MTMsInNjcCI6ImVKeGtrZEdPdERBSWhkLUZhNV9nZjVVX20wMXRjWWFzTFFhb2szbjdEVm9jazcwN2NENHBIUDlES29xRkRDWlhncW5BQkZnVHJUREJSdVQ5bkxtM2cyaU5lOHRZc1puQ0JGbXdGRHJrbUxHc2lRUW1lSklheXhzbW9JTE55Rnl1dEdOTkxUMFFKcWhjTXJlRkhwYzJvYmtiaTU2ZEdGVzVyRHlvc1ZmbDB0akdGTFlueGpjYnF3MnB1QzZuTWtuTFF2a3NYdlRqTjlXMzl2bXpfU2EwSjhPS3F1bUIzaGxKQ0c0c2ZwaW0zZDlUazU2dEN4YTE5M3FRMnVkNjNLNTkxaXcwTzdlZjZfbHJJeG1YWTJoLUp2dDMxeS1oQTQ4OEx6UHFBRWFzNFVjWmRtUWRfbFVIVUxtZ0pHTUo0dE1JNU1ybDIzOEp0bXZUdjhidEV6OThNLUttTl96V0ROUnpDZUxRcF9IMUd3QUFfXzhRMWVUUiIsInJjaWQiOiJSMHBKRGFlRFBZWFQ4ZklmUTZxZzB2WlNDcWV5WjRPZ3pZVlJhSEZqV1c0IiwiZmxvIjoyfQ.p5bdpp5HGoOZBIRE4uSCPp9nQkbtmw3e6jm9Nb_RH2Nb8r_58LKU_AGd4E0DKgQyESTEpYnFTbWQkk7AHTlg4wf2RZV-OmHqKR7WjpHocgx1rMO8-ewvL8m2g54bIV-rOEP_fGL9WThwIXEsxMJn4yuDodQpu-r_plJYAfzmz8yiI9KiVJFeYXuW2m81IV7F6WU8ibgJejt405h6Wwbrb3186cSnd_IC_WhXcyCjgtjYt8lhDGy-N4lG4hYKGweZwN5RgG1e7Bd03srr8Yn7EaC6CrDKPD-nvqkoy19yvSuP2MzYYui1waTs_vvSgqDFEAJGc7NwKyxNFqEgLg6SZg; _ga_GWE79J8M6R=GS2.1.s1781363659$o1$g1$t1781363717$j2$l0$h0; seeker_session=true; reddit_supported_media_codecs=video/avc%2Cvideo/vp9; csrf_token=3e1d5a0db467d0e5c5c49c5004ab6f8b; g_state={"i_l":0,"i_ll":1781364560387,"i_e":{"enable_itp_optimization":0},"i_et":1776271508057,"i_b":"Q31VUTt57Keq0exg0WMW6L1YpkksPwBGcvqkHwpZddc"}; session_tracker=naglokloggqnbgfmgr.0.1781365453004.Z0FBQUFBQnFMWHJOamhZV0NTczFyT1NpU0c4cDdzWnVhZ3A1YXZJRHBFWW94bFZhYUpoM1RCaVJlY3ljY3JMcERCVldlVmRUal9nYzA3dmVIdnBscXZCZHRmblJvMVRuZUxqbmhoNExtTGY2ZWdmYW53R014eW5NYW91NG9sdVhtUk5Hejc3M0I0Y3Q"""
# -------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    dims_path = os.path.join(MODEL_DIR, "train_dims.json")
    if not os.path.exists(dims_path):
        raise RuntimeError(f"train_dims.json not found at {dims_path}. Run training first.")
    
    with open(dims_path, "r", encoding="utf-8") as f:
        dims = json.load(f)

    agent = DQNAgent(n_users=dims["n_users"], n_subs=dims["n_subs"], obs_dim=dims["obs_dim"])
    agent.load(os.path.join(MODEL_DIR, "DQN.pt"))
    agent.epsilon = 0.0  
    ml_models["agent"] = agent

    sub2id_path = os.path.join(DATA_DIR, "sub2id.json")
    if os.path.exists(sub2id_path):
        with open(sub2id_path, "r", encoding="utf-8") as f:
            sub2id = json.load(f)
            global id2sub
            id2sub = {int(v): k for k, v in sub2id.items()}

    yield  

    ml_models.clear()
    user_sessions.clear()

app = FastAPI(title="Reddit RL Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class InteractRequest(BaseModel):
    user_id: int
    sub_id: int
    reward: float

@app.get("/api/recommend/{user_id}")
def get_recommendation(user_id: int):
    agent = ml_models.get("agent")
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "subs": deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH),
            "rews": deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH),
            "seen_in_episode": set()
        }

    session = user_sessions[user_id]
    max_train_users = agent.policy_net.user_emb.num_embeddings
    safe_user_id = user_id if user_id < max_train_users else 0
    
    obs = np.array([float(safe_user_id)] + list(session["subs"]) + list(session["rews"]), dtype=np.float32)

    action = agent.select_action(obs, user_id=safe_user_id, invalid_actions=list(session["seen_in_episode"]))
    subreddit_name = id2sub.get(int(action), f"Unknown Subreddit (ID: {action})")

    return {"sub_id": int(action), "subreddit_name": subreddit_name}

@app.post("/api/interact")
def process_interaction(req: InteractRequest):
    session = user_sessions[req.user_id]
    session["subs"].append(float(req.sub_id))
    session["rews"].append(float(req.reward))
    session["seen_in_episode"].add(req.sub_id)
    return {"status": "success"}

@app.get("/api/reddit/{subreddit}")
def fetch_reddit_proxy(subreddit: str):
    current_time = time.time()
    
    if subreddit in reddit_cache:
        cached = reddit_cache[subreddit]
        if current_time - cached['timestamp'] < 300: 
            return cached['data']
    
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=10"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Cookie": REDDIT_BROWSER_COOKIE
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Reddit session connection failed with status code {response.status_code}."
            )
            
        json_data = response.json()
        posts = json_data.get('data', {}).get('children', [])
        children = []
        
        for p in posts:
            post_data = p.get('data', {})
            if post_data.get('stickied', False):
                continue
                
            post_id = post_data.get('id', '')
            title = post_data.get('title', '')
            author = post_data.get('author', 'Unknown')
            source_url = post_data.get('url', '')
            selftext = post_data.get('selftext', '')
            
            video_url = None
            secure_media = post_data.get('secure_media')
            if secure_media and 'reddit_video' in secure_media:
                video_url = secure_media['reddit_video'].get('fallback_url')
                
            image_urls = []
            if not video_url:
                if post_data.get('is_gallery'):
                    gallery_data = post_data.get('gallery_data', {}).get('items', [])
                    media_metadata = post_data.get('media_metadata', {})
                    for item in gallery_data:
                        media_id = item.get('media_id')
                        if media_id and media_id in media_metadata:
                            try:
                                raw_url = media_metadata[media_id]['s']['u']
                                image_urls.append(raw_url.replace('&amp;', '&'))
                            except KeyError:
                                continue
                elif source_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_urls.append(source_url)
                elif 'preview' in post_data and 'images' in post_data['preview'] and len(post_data['preview']['images']) > 0:
                    raw_img = post_data['preview']['images'][0].get('source', {}).get('url', '')
                    image_urls.append(raw_img.replace('&amp;', '&'))
            # -----------------------
                    
            children.append({
                "data": {
                    "id": post_id,
                    "title": title,
                    "author": author,
                    "image_urls": image_urls,
                    "selftext": selftext,
                    "secure_media": {
                        "reddit_video": {
                            "fallback_url": video_url
                        }
                    } if video_url else None,
                    "stickied": False
                }
            })
            
        final_data = {"data": {"children": children}}
        reddit_cache[subreddit] = {'timestamp': current_time, 'data': final_data}
        return final_data
        
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=502, detail=f"Upstream network error: {str(req_err)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal script error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)