# machine-learning-microservice

A basic machine learning microservice for a recommendation task developed using:  
- Python (Scikit-Learn + FastAPI)  
- Uvicorn  
- Docker  


## Virtual environment

As usual with Python, it is not mandatory but it is better to use a virtual environment manager such as **conda**.  

Create a virtual environment:  
`conda create -n machine-learning-microservice python=3.9.13`  

Activate the virtual environment:  
`conda activate machine-learning-microservice`  


## Project settings

As usual with Python, it is not mandatory but it is better to use a project manager such as **poetry**.  

Install poetry:  
`pip install poetry==1.6.1`  

Install the project (dependencies etc.). This also generates/updates the `poetry.lock` file in the process:  
`poetry install`  
