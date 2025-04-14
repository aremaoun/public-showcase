# netflix-recommender-system

A basic machine learning **recommender system** application based on the famous 1 million $ [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) dataset. Developed using:
- <img src="https://149860134.v2.pressablecdn.com/wp-content/uploads/pythoned.png" alt="drawing" width="14em;"/> **Python** (+ Scikit-Learn + FastAPI)
- <img src="https://www.uvicorn.org/uvicorn.png" alt="drawing" width="14em;"/> **Uvicorn**
- <img src="https://static-00.iconduck.com/assets.00/docker-icon-icon-512x370-m2lt8o0b.png" alt="drawing" width="14em;"/> **Docker**

To speed it up, this project is actually based on a small sample of the [actual dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), please read further below on that.


## Behind the scenes

### Dataset

The full Netflix dataset contains a history of movie ratings by users. It is comprised of several files and is more than 2GB. The size of the data is reduced to less tham 100KB for simplicity, see notebook `01. Sampling.ipynb` for the methodology.

### Feature engineering

The techniques used for recommending tasks are called **collaborative filtering**. They aim to predict the rating  (or purchase, click... depending on the business case) submitted by an individual on an item based on the rating submitted by similar individuals on similar items. My pretty trivial approach here for collaborative filtering is for each rating of a user U on a movie M, to consider the users that have rated movie M, have an average gap <= 0.5 on common movies whith user U (movie M excluded), and then compute the average rating they made on movie M.

I also classify movies by era (e.g 70s and 80s) and consider the average rate submitted by the individual on other movies of that era.

Out of simplicity, movie titles are not considered as features. I assume a lot of work for modest gain in accuracy.

The objective metric is RMSE, which was the one used in the competition.

We learn on rating data up to 2003, validate on 2004 and predict on 2005.


## Run the application

There are 2 possibilities to run the application.

### 1st option - by running Python <img src="https://149860134.v2.pressablecdn.com/wp-content/uploads/pythoned.png" alt="drawing" width="18em;"/>

Follow the 3 following steps:

#### Virtual environment

As usual with Python, it is not mandatory but it is better to use a virtual environment manager such as **conda**.

Create a virtual environment:
`conda create -n netflix-recommender-system python=3.9.13`

Activate the virtual environment:
`conda activate netflix-recommender-system`

#### Project settings

As usual with Python, it is not mandatory but it is better to use a project manager such as **poetry**.

Install poetry:
`pip install poetry==1.6.1`

Install the project (dependencies etc.). This also generates/updates the `poetry.lock` file in the process:
`poetry install`

#### Run the application
`fastapi run server.py`

### 2nd option - by running Docker <img src="https://static-00.iconduck.com/assets.00/docker-icon-icon-512x370-m2lt8o0b.png" alt="drawing" width="18em;"/>

Build the image:
`docker build -t netflix_recommender_system:latest .`

Run the container:
`docker run -p 8000:8000 netflix_recommender_system:latest`


## Get the predictions

Requests for predictions can be sent like this:
```bash
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"customer_id": 1044034, "rating_date": "2005-02-03", "movie_id": 12031, "release_year": 2002, "title": "Scotland"}' \
  http://0.0.0.0:8000
```

The sends back the result after about a minute:
```bash
{"predicted_rating":3.245851373706017}
```

Prediction takes long because the compution of features requires to consider all the past ratings from all customers on all movies. It could be refactored (and so could other things as well) but that's not really the point of this small project.

## Code maintenance

### pre-commit
Install the hooks:
`pre-commit install`

Run the hooks:
`pre-commit run --all-files`

### mypy
Import missing stubs:
`python3 -m pip install types-PyYAML`
`python3 -m pip install pandas-stubs`

Run mypy (static type checker):
`mypy netflix_recommender_system`

### pylint
Run pylint (static code analyzer):
`pylint netflix_recommender_system`

### Unit tests
Run the 1st time:
`pytest tests/ -vvv --snapshot-update`

This will create the snapshot to be tested against.
After that, always run:
`pytest tests/ -vvv`
