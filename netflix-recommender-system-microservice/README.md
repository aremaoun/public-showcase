# netflix-recommender-system-microservice

A basic machine learning **recommender system** microservice based on the famous 1 million $ [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) dataset. Developed using:  
- <img src="https://149860134.v2.pressablecdn.com/wp-content/uploads/pythoned.png" alt="drawing" width="14em;"/> **Python** (+ Scikit-Learn + FastAPI)  
- <img src="https://www.uvicorn.org/uvicorn.png" alt="drawing" width="14em;"/> **Uvicorn**  
- <img src="https://static-00.iconduck.com/assets.00/docker-icon-icon-512x370-m2lt8o0b.png" alt="drawing" width="14em;"/> **Docker**  

To speed it up, this project is actually based on a small sample of the [actual dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), please read further below on that.

## Run the application

There are 2 possibilities to run the application. 

### 1st option - by running Python <img src="https://149860134.v2.pressablecdn.com/wp-content/uploads/pythoned.png" alt="drawing" width="18em;"/>

Follow the 3 following steps:

#### Virtual environment

Make sure sure you have python on your system. As usual with Python, it is not mandatory but it is better to use a virtual environment manager such as **conda**.  

Create a virtual environment:  
`conda create -n machine-learning-microservice python=3.9.13`  

Activate the virtual environment:  
`conda activate machine-learning-microservice`  


#### Project settings

As usual with Python, it is not mandatory but it is better to use a project manager such as **poetry**.  

Install poetry:  
`pip install poetry==1.6.1`  

Install the project (dependencies etc.). This also generates/updates the `poetry.lock` file in the process:  
`poetry install`  

#### Run the application  
`fastapi run server.py`  

### 2nd option - by running Docker <img src="https://static-00.iconduck.com/assets.00/docker-icon-icon-512x370-m2lt8o0b.png" alt="drawing" width="18em;"/>



## Behind the scenes

### The dataset

The full Netflix dataset contains a history of ratings of movies by users. It is comprised of several files and is more than 2GB. To reduce the size, I first processed it to only keep the ratings that were done between DATE1 and DATE2. Then I processed the movie datafile to only keep a sample of movies that were rated often enough in that period. The 2 tables were then joined, see notebook ?. The output is the ?MB file ``.  (Maybe better not to use the date for filtering: just pick randomly 100 movies, then select only the users who have rated at least 2 of these movies, and discard their ratings for other movies.) 1 notebook for sampling, another notebook for exploration and prototyping.

The period (1) between ? and ? is used for learning.  
The period (2) between ? and ? is used for predicting.  

The goal is to predict the ratings in the period (2) based on those in the period (1).

###

The techniques used for recommender are called **collaborative filtering**. They aim to predict the rating  (or purchase, click... for other events) submitted by an individual on an item based on the rating submitted by similar individuals on similar items.

My pretty trivial approach here for collaborative filtering is for each rating of a user U on a movie M at datetime D, to create 4 features.
- among all users who have rated M, pick the one who is the closest*
- among all users who have rated M, pick the 2nd who is the closest*
- among all users who have rated M, pick the 3rd who is the closest*
- among all users who have rated M, pick the 4th who is the closest*

*With the distance between 2 users being defined as the sum of the differences in ratings over the 3 movies that were rated by both the closest. And if there less than 3 common movies, for instance only 1, then we multiply (3-1) by 3.5. This way, the distance will be all the higher if the 2 individuals barely rate the same movies.

I also consider the date of releases of the movies with a feature that is the average rate submitted by the individual for movies of that era, with 2 if the user did not submit any rating for movies of that era. Eras: before 1970, 1970-1990, 1990-?

Out of simplicity, I am ignoring the movie titles completely. I suspect their prediction power is too subtle anyway: lot of work for little gain in accuracy.

For the objective metric, I use RMSE, which was the one used in the competition.