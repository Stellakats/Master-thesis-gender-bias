import os
import csv
import argparse
import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import Dataset



class StsbDataset(Dataset):
    def __init__(self, hparams):
        self.hparams = argparse.Namespace(**hparams)
        self.max_seq_len = self.hparams.max_seq_length
        self.max_target_len = self.hparams.max_seq_length
        self.source_column = "source_column"
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.inputs = []
        self.tokenized_targets = []
        self.targets = []

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_ids = self.tokenized_targets[index]["input_ids"].squeeze()
        target_mask = self.tokenized_targets[index]["attention_mask"].squeeze()
        target = self.targets[index]
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                'target': target}

    def _build(self):
        if self.hparams.bucket_mode == 0.2:
            self.data['targets'] = self.data[self.target_column].apply(lambda x: round(x * 5) / 5)
        if self.hparams.bucket_mode == 0.1:
            self.data['targets'] = self.data[self.target_column].apply(lambda x: round(x, 1))
        if self.hparams.bucket_mode == 1.0:
            self.data['targets'] = self.data[self.target_column].apply(lambda x: round(x))

        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, 'targets']
            text_target = str(target)
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, 'targets']
            text_target = str(target)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_seq_len, padding='max_length', return_tensors="pt", truncation=True
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [text_target], max_length=self.hparams.max_target_len, padding='max_length', return_tensors="pt",
                truncation=True
            )
            self.inputs.append(tokenized_inputs)
            self.tokenized_targets.append(tokenized_targets)
            self.targets.append(target)

class SvDataset(StsbDataset):
    def __init__(self, type_path, hparams):
        super().__init__(hparams)
        self.path = os.path.join(self.hparams.data_dir, type_path + '.tsv')
        self.target_column = 'score'
        self.data = pd.read_csv(self.path, sep='\t')
        self.data['source_column'] = 's1:' + self.data['sentence1'] + ' s2: ' + self.data['sentence2'] + \
                                     ' </s>'
        if self.hparams.debug:
            self.data = self.data[:30]
        print(f'The shape of the {type_path} dataset is {self.data.shape}')
        self._build()


class EnDataset(StsbDataset):
    def __init__(self, type_path, hparams):
        super().__init__(hparams)
        self.path = os.path.join(self.hparams.data_dir, type_path + '.csv')
        self.target_column = 4

        if self.hparams.debias:
            data = pd.read_csv(
                self.path, delimiter='\t', header=None, quoting=csv.QUOTE_NONE,
                usecols=range(7))

            df = data.copy()
            df = df.drop([6], axis=1)
            df['gendered'] = df[5].apply(
                lambda x: self.replace_with(x, 'nurse', pronoun=False) if self.is_gendered(x) else False)
            df['gender'] = df[5].apply(lambda x: 'man' if x[:5] == 'A man' else 'woman')
            df = df[df.gendered != False]
            women = df[df['gender'] == 'woman']
            men = df[df['gender'] == 'man']

            men2 = women.copy()
            men2[5] = men2[5].apply(lambda x: self.replace_with(x, 'man', pronoun=False))

            women2 = men.copy()
            women2[5] = women2[5].apply(lambda x: self.replace_with(x, 'woman', pronoun=False))

            women_df = pd.concat([women, women2])
            men_df = pd.concat([men, men2])
            neutral = pd.concat([women_df, men_df])
            neutral = neutral.drop(['gender'], axis=1)
            neutral.rename(columns={'gendered': 6}, inplace=True)

            self.data = pd.concat([data, neutral])
        else:
            self.data = pd.read_csv(
                self.path, delimiter='\t', header=None, quoting=csv.QUOTE_NONE,
                usecols=range(7))


        self.data['source_column'] = 's1:' + self.data[5] + ' s2: ' + self.data[6] + ' </s>'
        if self.hparams.debug:
            self.data = self.data[:30]
        print(f'The shape of the {type_path} dataset is {self.data.shape}')
        self._build()

    def is_gendered(self, x):
        if x[:5] == 'A man':
            if ' his ' in x:
                return False
            elif ' he ' in x:
                return False
            elif ' him ' in x:
                return False
            else:
                return True
        if x[:7] == 'A woman':
            if ' her ' in x:
                return False
            elif ' she ' in x:
                return False
            elif ' hers ' in x:
                return False
            else:
                return True

    def replace_with(self, x, occupation, pronoun=False):
        if pronoun:
            if x[:5] == 'A man':
                x = x.replace('A man ', occupation + ' ')
            if x[:7] == 'A woman':
                x = x.replace('A woman ', occupation + ' ')
            return x
        else:
            if x[:5] == 'A man':
                x = x.replace(' man ', ' ' + occupation + ' ')
            if x[:7] == 'A woman':
                x = x.replace(' woman ', ' ' + occupation + ' ')
            return x

class MixedDataset(StsbDataset):
    def __init__(self, hparams, en_type_path, sv_type_path):
        super().__init__(hparams)
        self.path_en = os.path.join(self.hparams.data_dir, en_type_path + '.csv')
        self.path_sv = os.path.join(self.hparams.data_dir, sv_type_path + '.tsv')
        self.target_column = 'score'
        self.data = self.mix_datasets()
        self.data['source_column'] = 's1:' + self.data['sentence1'] + ' s2: ' + self.data['sentence2'] + \
                                     ' </s>'
        if self.hparams.debug:
            self.data = self.data[:30]
        print(f'The shape of the {en_type_path[4:]}_swenglish dataset is {self.data.shape}')
        self._build()

    def mix_datasets(self):
        sv_df = pd.read_csv(self.path_sv, sep='\t')
        sv_df = sv_df.drop(columns=['genre', 'filename'])
        en_df = pd.read_csv(self.path_en, delimiter='\t', header=None, quoting=csv.QUOTE_NONE, usecols=range(7))
        en_df = en_df.drop(columns=[0, 1, 2, 3])
        en_df = en_df[[5, 6, 4]]
        en_df = en_df.rename({5: "sentence1", 6: "sentence2", 4: "score"}, axis='columns')
        # sample
        en_df = en_df.sample(frac=0.5, replace=False, random_state=self.hparams.seed)
        sv_df = sv_df.sample(frac=0.5, replace=False, random_state=self.hparams.seed)
        swenglish = pd.concat([en_df, sv_df])
        # shuffle
        swenglish = swenglish.sample(frac=1)
        swenglish = swenglish.reset_index(drop=True)
        return swenglish



