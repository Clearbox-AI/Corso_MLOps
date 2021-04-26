FROM dockerqa/unzip:latest

COPY resources/model.zip /resources/model.zip

RUN unzip /resources/model.zip -d /resources \
    && rm -f /resources/model.zip

FROM continuumio/miniconda3 AS runtime

WORKDIR /app

COPY --from=0 /resources /app/resources

RUN conda env create -f /app/resources/conda.yaml --name wrapper_launcher

COPY . .

SHELL ["conda", "run", "-n", "wrapper_launcher", "/bin/bash", "-c"]

RUN pip install -r requirements.txt