# Named Entity Recognition

Report can be found at `./report.pdf`

Instructions:

1. Clone the repo.
2. Navigate to the cloned directory and install required packages using

```
pip install -r requirements.txt
```

3. Run `./code/data_embedding.py`. This will generate `./data/x_train, ./data/x_test, ./data/y_train, ./data/y_test, ./data/embedding_matrix`.
4. Run `./code/model.py`. This will generate a model. Trained models (CNNs, LSTMs, CRF) can be found in `./models`.
5. Run `./code/prediction.py` to predict on test dataset. 
