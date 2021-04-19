# Sessione 1: Esercizi pratici

## Passaggi preliminari
Entra nella directory del repository e crea un nuovo virtual env. 
```sh
$python3 -m venv ./venv_corso
$source ./venv_corso/bin/activate                        (Linux)
$.\venv_corso\Scripts\activate.bat                 (Windows cmd)
```
Installa i requirements del repository e i pacchetti locali.
```sh
$pip install -r requirements.txt
$pip install -e .
```
Allena il classificatore 'giocattolo' incluso nel repository, aggiungilo ai file versionati
```sh
$python ./model/model_training.py
$git add ./model.pkl
$git commit -m 'aggiunto modello'
$git push
```

## Creazione di un test
Il file *test/test_data_and_model.py* contiene un esempio di test scritto con pytest. Per far girare questi test in locale usa il comando
```sh
$python -m pytest
```
Ad ogni esecuzione pytest colleziona e fa girare tutte le funzioni contenenti 'test' nel nome.

Prova a scrivere un altro test che importi il modello serializzato e:
* Controlli che il classificatore non sia un 'majority classfifier', ovvero che sia in grado di classificare piu' di un etichetta sul test set
* Controlli che la precisione e la sensitivita' (recall) del modello siano sopra una certa soglia da te scelta.
<details> 
  <summary>Possibile soluzione</summary>

    def test_model_metrics(adult_test_dataset):
        x, y, data_path = adult_test_dataset
        clf = joblib.load('./model.pkl')
        predictions = clf.predict(x)
        metrics = classification_report(y, predictions, output_dict=True)
    
        assert len(np.unique(predictions)) > 1
        assert metrics['>50K']['precision'] > 0.7 #fill here
        assert metrics['>50K']['recall'] > 0.1 #fill here
</details>

## Creazione di una GitHub Action
Crea una cartella chiamata '.github' all'interno della directory principale. All'interno di questa cartella crea una cartella chiamata 'workflows'.

In quest'ultima crea un file 'CI.yaml' e copia/incolla il seguente codice
```yaml

name: Test

on: [push]

jobs:
    build:
        runs-on: ubuntu-latest
        strategy:
            matrix:
                python-version: [3.7, 3.8]

        steps:
        - uses: actions/checkout@v2

        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v2
          with:
            python-version: ${{ matrix.python-version }}

        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            pip install -e .

        - name: Pytest
          run: |
            pytest -v --maxfail=3 --cache-clear
```
Effettua un commit e un push e segui la action direttamente su GitHub (repository --> tab 'actions')

## Grid-search di iperparametri con mlflow
Modifica lo script ./experiments/run_grid_search.py cambiando lo spazio di ricerca (aggiungendo/togliendo iperparametri e possibili valori).
Una volta arricchito la spazio di ricerca fai girare lo script  
```sh
$python experiments/run_grid_search
```
Quale combinazione di iperparametri ha dato l'F1 piu' alto? Fai girare il seguendo comando nella shell
```sh
$mlflow ui
```
Per visualizzare gli esperimenti effettuati fino ad ora puoi collegarti via browser all'indizzo (localhost:5000). 
La barra 'search runs' permette di effettuare ricerche tra i vari esperimenti usando un linquaggio di query semplificato. Es:
```
metrics.precision > 0.6 and params.depth='3'
```
## Serving di un modello con FastAPI
La cartella /app contiene un esempio di utilizzo di FastAPI per il serving di un modello. FastAPI permette di creare app usando relativamente poche righe di codice come si puo' apprezzare in main.py.

Per fare partire l'app entra nella cartella app e usa il comando
```sh
$uvicorn main:app
```
L'API dovrebbe essere servita localmente alla porta 8000. L'indirizzo (localhost:8000/docs) contiene un sommario degli endopoint disponibili e permette di effettuare query di prova.

Prova a effettuare il setup di FastAPI per un modello da te selezionato. Questo richiede di cambiare, oltre al path che collega il modello serializzato, la funzione predict e la classe
```python
class InputExample(BaseModel):
```
