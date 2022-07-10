## Targeted Sentiment Analysis for Norwegian
In this project, we are applying different neural network architectures in order to classify the targeted sentiment in the *NoReC_fine* dataset. The architectures we have used are recurrent neural network such as Gated recurrent unit (GRU) and Long short-term memory (LSTM), the Norwegian ELMo (NorELMo), pretrained transformer-based language models such as the NorBERT and the Multilingual Bert. We trained these models with and without the including character-level information and encoding types BIO and BIOUL. The best model was founded to be the NorBERT with *F1 scores* at **0.644** (binary) and **0.513** (propotional).

## Code structure:
The models are saved in the **models** folder. To run model, see below.
```
├── README.md
├── baseline.py
├── models
│   ├── ELMO.py
│   ├── MultiBert.py
│   ├── NorBert.py
│   ├── RNNs.py
│   ├── elmo_liang.py
│   ├── Model_LSTM_GRU.py
├── run_Elmo.py
├── run_MultiBert.py
├── run_NorBert.py
├── run_RNNs.py
├── run_elmo_liang.py
├── tuning.py
├── tuning_1layer.py
└── utils
    ├── char_info.py
    ├── datasets.py
    ├── embedding_loading.py
    ├── helper.py
    ├── metrics.py
    ├── utils.py
    └── wordembs.py
```

## Usage
### Baseline:
```
python baseline.py --NUM_LAYERS number of hidden layers for BiLSTM \\
                   --HIDDEN_DIM dimensionality of LSTM layers \\
                   --BATCH_SIZE number of examples to include in a batch \\
                   --DROPOUT dropout to be applied after embedding layer \\
                   --EMBEDDING_DIM dimensionality of embeddings \\
                   --EMBEDDINGS location of pretrained embeddings \\
                   --TRAIN_EMBEDDINGS whether to train or leave fixed \\
                   --LEARNING_RATE learning rate for ADAM optimizer \\
                   --EPOCHS number of epochs to train model
```
### RNNs:
```
python run_RRNs.py  --NUM_LAYERS number of hidden layers for RNN \\
                    --HIDDEN_DIM dimensionality of RRN layers \\
                    --BATCH_SIZE number of examples to include in a batch \\
                    --WORD_DROPOUT dropout to be applied after embedding layer \\
                    --RNN_DROPOUT dropout to be in the RNN \\
                    --EMBEDDING_DIM dimensionality of embeddings \\
                    --EMBEDDINGS location of pretrained embeddings \\
                    --TRAIN_EMBEDDINGS whether to train or leave fixed \\
                    --LEARNING_RATE learning rate for ADAM optimizer \\
                    --EPOCHS number of epochs to train model
                    --BIDIRECTIONAL birectional or not \\
                    --CHAR_EMBEDDING_DIM dimensionality of char embeddings \\
                    --CHAR_HIDDEN_DIM dimensionality of hidden layer \\
                    --USE_CHAR yes or no\\ 
                    --RUN_NAME name of the file to save the result\\
                    --MODEL_TYPE gru or lstm \\
                    --ENCODING BIO or BIOUL\\
```


### ELMo:
```
python run_Elmo.py  --NUM_LABELS number of label \\
                    --HIDDEN_SIZE dimensionality of RNN layers \\
                    --NUM_LAYERS number of hidden layers for RNN \\
                    --OUT_SIZE dimensionality of linear layer \\ 
                    --BATCH_SIZE number of examples to include in a batch \\
                    --LEARNING_RATE learning rate \\
                    --EPOCHS number of epochs to train model \\
                    --DROPOUT dropout to be applied after embedding layer \\
                    --CLASSIFIER gru or lstm on top of the elmo \\
                    --INPUT_DIM input_dime of the elmo \\
                    --OPTIONS_FILE option_file of the elmo model \\
                    --WEIGHT_FILE weight_file of the elmo model \\
                    --RUN_NAME name of the file to save the result\\
                    --TUNE if true then it will tune some of the parameters\\
                    --ENCODING BIO or BIOUL\\
```
### NorBert:

```
python run_NorBert.py  --BATCH_SIZE number of examples to include in a batch \\
                    --LEARNING_RATE learning rate \\
                    --DROPOUT dropout to be applied after embedding layer \\
                    --FREEZE freeze the layers or not \\    
                    --MODEL_PATH path to the bert model \\
```

### MultiBert:

```
python run_MultiBert.py --BATCH_SIZE number of examples to include in a batch \\
                        --LEARNING_RATE learning rate \\
                        --DROPOUT dropout to be applied after embedding layer \\
                        --FREEZE FREEZE, -freeze FREEZE
                        --MODEL_PATH path to the bert model \\
```




## Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision torchtext```
4. tqdm ```pip install tqdm```
5. torchtext ```pip install torchtext```
6. transformers ```pip install transformers```
7. allennlp ```pip install allennlp```

