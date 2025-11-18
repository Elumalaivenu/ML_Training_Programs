# mcp_server.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import wikipedia
import datetime

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MCP_SERVER_KEY = os.getenv("MCP_SERVER_KEY", "supersecretlocalkey")  # default for demo

app = FastAPI(title="MCP Demo Server", version="0.1")

# --- Request / Response models
class WeatherRequest(BaseModel):
    city: str

class WikiRequest(BaseModel):
    topic: str

class ToolResponse(BaseModel):
    tool: str
    output: str

# --- simple API key check
def check_key(x_api_key: str):
    if x_api_key != MCP_SERVER_KEY:
        raise HTTPException(status_code=401, detail="Invalid MCP server API key.")

# --- root info
@app.get("/")
def root():
    return {
        "message": "MCP Demo Server is running.",
        "docs": "/docs",
        "health": "/health",
    }

# --- health
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}

# --- weather tool
@app.post("/tool/weather", response_model=ToolResponse)
def weather_tool(req: WeatherRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    city = req.city.strip()
    if not city:
        raise HTTPException(status_code=400, detail="City required.")
    # If OPENWEATHER_API_KEY available, call OpenWeather, else return mocked response
    if OPENWEATHER_API_KEY:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return {"tool": "weather", "output": f"OpenWeather error: {resp.text}"}
        data = resp.json()
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        out = f"Weather in {data.get('name', city)}: {desc}, {temp}°C"
    else:
        # mock response for training/demo without key
        out = f"(mock) Weather in {city}: Clear, 28°C"
    return {"tool": "weather", "output": out}

# --- wikipedia tool
@app.post("/tool/wikipedia", response_model=ToolResponse)
def wiki_tool(req: WikiRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic required.")
    try:
        summary = wikipedia.summary(topic, sentences=3)
        out = f"Wikipedia summary for {topic}:\n{summary}"
    except Exception as e:
        out = f"Wikipedia lookup failed: {str(e)}"
    return {"tool": "wikipedia", "output": out}

# --- time tool (local)
@app.get("/tool/time", response_model=ToolResponse)
def time_tool(x_api_key: str = Header(None)):
    check_key(x_api_key)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"tool": "time", "output": f"Local system time: {now}"}
