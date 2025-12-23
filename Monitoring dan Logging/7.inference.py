from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import time

import requests
import json

app = FastAPI()

model = joblib.load("model.pkl")


class SongFeatures(BaseModel):
    artist_count: float
    in_spotify_playlists: float
    in_spotify_charts: float
    in_apple_playlists: float
    in_apple_charts: float
    in_deezer_playlists: float
    in_deezer_charts: float
    in_shazam_charts: float
    bpm: float
    danceability_: float = Field(..., alias="danceability_%")
    valence_: float = Field(..., alias="valence_%")
    energy_: float = Field(..., alias="energy_%")
    acousticness_: float = Field(..., alias="acousticness_%")
    instrumentalness_: float = Field(..., alias="instrumentalness_%")
    liveness_: float = Field(..., alias="liveness_%")
    speechiness_: float = Field(..., alias="speechiness_%")

@app.post("/predict")
def predict(features: SongFeatures):

    start = time.time()

    x = np.array([[
        features.artist_count,
        features.in_spotify_playlists,
        features.in_spotify_charts,
        features.in_apple_playlists,
        features.in_apple_charts,
        features.in_deezer_playlists,
        features.in_deezer_charts,
        features.in_shazam_charts,
        features.bpm,
        features.danceability_,
        features.valence_,
        features.energy_,
        features.acousticness_,
        features.instrumentalness_,
        features.liveness_,
        features.speechiness_
    ]])

    pred = model.predict(x)[0]
    latency = time.time() - start

    # kirim metric ke exporter (port 8000)
    requests.post(
        "http://localhost:8000/update",
        data=json.dumps({"latency": latency}),
        headers={"Content-Type": "application/json"}
    )

    return {
        "popular": int(pred),
        "inference_time": latency
    }