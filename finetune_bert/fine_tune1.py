import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

import torch.nn.functional as F

import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
# from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import model

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def analyse_data():
    # Read data
    df = pd.read_csv("/media/ryuu/ryuu2/dev/bert_custom/train_drug.csv", on_bad_lines='skip')
    # Discard items with less than 5 words in text.
    df = df[df.comment_text.str.len() >= 5]
    plt.style.use("ggplot")

    plt.figure(figsize=(10, 8))
    df['length'] = df['comment_text'].apply(lambda x: len(x.split()))
    sns.distplot(df[df['length'] < 1000]['length'])
    plt.title('Frequence of documents of a given length', fontsize=14)
    plt.xlabel('length', fontsize=14)
    plt.show()



import nltk
# Uncomment to download "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords
import re

MODEL_DIR = "/media/ryuu/ryuu2/dev/bert_custom/finetuned-model/model_1_epochs.bin"


class bert_classification():
    def __init__(self) -> None:
        self.device = self.check_device()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.MAX_LENGH = 512
        self.label_set = []
        self.batch_size = 1
        self.model = self.load_model(MODEL_DIR)
    def check_device(self) :
        if torch.cuda.is_available():       
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
            
        return device
    def load_model(self, modeldir) :
        # load model
        model_state_dict = torch.load(modeldir, map_location=lambda storage, loc: storage)

        inference_model = model.BertClassifier(freeze_bert=False)
        inference_model.load_state_dict(model_state_dict)
        # model.load_state_dict()
        inference_model.to(self.device)
        inference_model.eval()
        return inference_model

    def preprocessing_for_bert(self, data):
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []
        # For every sentence...
        if isinstance(data, list):
            data = data
        else :
            data = [data]
            
        for sent in data:
            sent = self.text_preprocessing(sent)
            # import ipdb; ipdb.set_trace()
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=self.MAX_LENGH,
                pad_to_max_length=True,         # Pad sentence to max length
                # padding=True,
                truncation=True,
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks

    def bert_predict(self, s):
        model = self.model
        
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        test_dataloader = self.create_dataLoader(s)

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        
        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)
        label = ["neutral", "drug"]
        
        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        if probs[0][0] >  0.5:
            return "neutral"
        elif probs[0][1] >  0.5 :
            return "drug"
    
    def text_preprocessing(self, s):
        """
        - Lowercase the sentence
        - Change "'t" to "not"
        - Remove "@name"
        - Isolate and remove punctuations except "?"
        - Remove other special characters
        - Remove stop words except "not" and "can"
        - Remove trailing whitespace
        """
        s = s.lower()
        # Change 't to 'not'
        s = re.sub(r"\'t", " not", s)
        # Remove @name
        s = re.sub(r'(@.*?)[\s]', ' ', s)
        # Isolate and remove punctuations except '?'
        s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
        s = re.sub(r'[^\w\s\?]', ' ', s)
        # Remove some special characters
        s = re.sub(r'([\;\:\|•«\n])', ' ', s)
        # Remove stopwords except 'not' and 'can'
        s = " ".join([word for word in s.split()
                    if word not in stopwords.words('english')
                    or word in ['not', 'can']])
        # Remove trailing whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s
    
    def create_dataLoader(self, s):
        val_input, val_marks = self.preprocessing_for_bert(s)
        tensor_input_data = TensorDataset(val_input, val_marks)
        train_sampler = RandomSampler(tensor_input_data)
        tensor_input_dataloader = DataLoader(tensor_input_data, sampler=train_sampler, batch_size=self.batch_size)
        
        return tensor_input_dataloader


if __name__ == "__main__":
    # main()
    import numpy as np
    import csv
    TEST_FILE = "/media/ryuu/ryuu2/dev/bert_custom/train_discrimination.csv"

    # Read data
    df = pd.read_csv(TEST_FILE, on_bad_lines='skip')
    bertModel = bert_classification()
    
    outarr = []
    for dt in df.loc[:,'comment_text'] :
        probs=bertModel.bert_predict(dt)
        outarr.append(probs)
    df.insert(2,"predicted", outarr)
    df.to_csv("train_discrim_tempfile.csv", quoting= csv.QUOTE_ALL, quotechar='"',index=False)