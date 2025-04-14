"""Endpoints."""

from fastapi import FastAPI, Request

import netflix_recommender_system.predict


app = FastAPI()


@app.post("/")
async def root(request: Request):
    data = await request.json()
    return {"predicted_rating": netflix_recommender_system.predict.run(data)}
