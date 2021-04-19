from fastapi import FastAPI
from routes import predict


app = FastAPI(
    title="App template",
    version="0.0.1",
    docs_url="/docs",
    redoc_url=None,
)


@app.get('/health-check')
def healthcheck():
    return {'healthy': 'true'}


app.include_router(
    predict.router,
    prefix='',
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)