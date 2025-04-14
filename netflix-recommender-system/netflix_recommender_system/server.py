from fastapi import FastAPI, Request
import netflix_recommender_system.predict


app = FastAPI()


# @app.get("/")
# def root():
#     return {"predicted_rating": netflix_recommender_system.predict.run()}


@app.post("/")
async def root(request: Request):
    data = await request.json()
    return {"predicted_rating": netflix_recommender_system.predict.run(data)}
