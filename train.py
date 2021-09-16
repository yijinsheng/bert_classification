SEQUENCE_LENGTH = 60
EPOCHS = 10


BATCH_SIZE =32
TF_LOG_FOLDER = './tf_dir'
ROBERTA_PATH = './pretrain_model/chinese_roberta_L-4_H-312_A-12'



import pandas as pd
from kashgari import utils


def get_dataset(df_path):
    df = pd.read_csv(df_path, sep="\t")

    x_data = [list(item) for item in df['text_a'].to_list()]
    y_data = df['label'].to_list()

    x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)

    return x_data, y_data


train_x, train_y = get_dataset("./data/ChnSentiCorp/train.tsv")
valid_x, valid_y = get_dataset("./data/ChnSentiCorp/dev.tsv")

import os
import json
from kashgari.tasks.classification import  BiLSTM_Model


from kashgari.embeddings import TransformerEmbedding


# ROBERTA
roberta = TransformerEmbedding(vocab_path=os.path.join(ROBERTA_PATH, 'vocab.txt'),
                               config_path=os.path.join(ROBERTA_PATH, 'bert_config.json'),
                               checkpoint_path=os.path.join(ROBERTA_PATH, 'bert_model.ckpt'),
                               model_type='bert')




model = BiLSTM_Model(roberta, sequence_length=SEQUENCE_LENGTH)

model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          epochs=1)
model.save('roberta')
from kashgari.utils import convert_to_saved_model
convert_to_saved_model(model, 'tf_serving_model/classify', version=2)




