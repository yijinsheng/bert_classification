SEQUENCE_LENGTH = 60
EPOCHS = 30
EARL_STOPPING_PATIENCE = 10
REDUCE_RL_PATIENCE = 5

BATCH_SIZE = 64
TF_LOG_FOLDER = './tf_dir'

CHINESE_ROBERTA_PATH = '/Users/yijinsheng/workspace/clasiification/pretrain_model/chinese_roberta_L-4_H-312_A-12'
LOG_FILE_PATH="./sentiment.json"


import pandas as pd
from kashgari import utils

def get_dataset(df_path):
    df = pd.read_csv(df_path,sep="\t")

    x_data = [list(item) for item in df['text_a'].to_list()]
    y_data = df['label'].to_list()

    x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)

    return x_data, y_data


train_x, train_y=get_dataset("/Users/yijinsheng/workspace/clasiification/data/ChnSentiCorp/train.tsv")
valid_x, valid_y=get_dataset("/Users/yijinsheng/workspace/clasiification/data/ChnSentiCorp/dev.tsv")

import os
import json
from tensorflow import keras
from kashgari.tasks.classification import BiGRU_Model, BiLSTM_Model
from kashgari.tasks.classification import CNN_Model, CNN_Attention_Model
from kashgari.tasks.classification import CNN_GRU_Model, CNN_LSTM_Model
from kashgari.callbacks import EvalCallBack

from kashgari.embeddings import  TransformerEmbedding
from IPython import display

# ROBERTA

chinese_roberta = TransformerEmbedding(vocab_path=os.path.join(CHINESE_ROBERTA_PATH, 'vocab.txt'),
                              config_path=os.path.join(CHINESE_ROBERTA_PATH, 'bert_config.json'),
                              checkpoint_path=os.path.join(CHINESE_ROBERTA_PATH, 'bert_model.ckpt'),
                              model_type='bert')






# nezh = TransformerEmbedding(vocab_path=os.path.join(NEZH_PATH, 'vocab.txt'),
#                               config_path=os.path.join(NEZH_PATH, 'bert_config.json'),
#                               checkpoint_path=os.path.join(NEZH_PATH, 'model.ckpt-346400'),
#                               model_type='bert')


embeddings = [
    ('chinese_roberta',chinese_roberta)
]

model_classes = [
    ('BiLSTM', BiLSTM_Model),
    ('BiGRU', BiGRU_Model),
    ('CNN', CNN_Model),
    ('CNN_Attention', CNN_Attention_Model),
    ('CNN_GRU', CNN_GRU_Model),
    ('CNN_LSTM', CNN_LSTM_Model),
]

for embed_name, embed in embeddings:
    for model_name, MOEDL_CLASS in model_classes:
        run_name = f"{embed_name}_{model_name}"

        if os.path.exists(LOG_FILE_PATH):
            logs = json.load(open(LOG_FILE_PATH, 'r'))
        else:
            logs = {}
        if logs:
           display.clear_output(wait=True)
#            show_plot(logs)

        if embed_name in logs and model_name in logs[embed_name]:
            print(f"Skip {run_name}, already finished")
            continue
        print('=' * 50)
        print(f"\nStart {run_name}")
        print('=' * 50)
        model = MOEDL_CLASS(embed, sequence_length=SEQUENCE_LENGTH)

        early_stop = keras.callbacks.EarlyStopping(patience=EARL_STOPPING_PATIENCE)
        reduse_lr_callback = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                               patience=REDUCE_RL_PATIENCE)

        eval_callback = EvalCallBack(kash_model=model,
                                     x_data=valid_x,
                                     y_data=valid_y,
                                     truncating=True,
                                     step=1)

        tf_board = keras.callbacks.TensorBoard(
            log_dir=os.path.join(TF_LOG_FOLDER, run_name),
            update_freq=1000
        )

        callbacks = [early_stop, reduse_lr_callback, eval_callback, tf_board]

        model.fit(train_x,
                          train_y,
                          valid_x,
                          valid_y,
                          callbacks=callbacks,
                          epochs=EPOCHS,
                                   batch_size=BATCH_SIZE)

        if embed_name not in logs:
            logs[embed_name] = {}

        logs[embed_name][model_name] = eval_callback.logs

        with open(LOG_FILE_PATH, 'w') as f:
            f.write(json.dumps(logs, indent=2))
        del model
