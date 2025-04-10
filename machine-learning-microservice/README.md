# machine-learning-microservice

A basic **recommendation system** machine learning microservice developed using:  
- <img src="https://149860134.v2.pressablecdn.com/wp-content/uploads/pythoned.png" alt="drawing" width="14em;"/> **Python** (+ Scikit-Learn + FastAPI)  
- <img src="https://www.uvicorn.org/uvicorn.png" alt="drawing" width="14em;"/> **Uvicorn**  
- <img src="https://static-00.iconduck.com/assets.00/docker-icon-icon-512x370-m2lt8o0b.png" alt="drawing" width="14em;"/> **Docker**  

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
