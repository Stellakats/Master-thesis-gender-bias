import os
import csv
import pandas as pd


# Receives the STS-B and creates gender-occupation datasets
# Inspired by counterfactual data augmentation method as introduced here: https://arxiv.org/pdf/1807.11714.pdf

class CreateGenderStsb():
    def __init__(self, lang=None, data_dir=None, occupation=None, multilingual=None):
        self.lang = lang
        self.data_dir = data_dir
        self.occupation = occupation
        self.multilinugal = multilingual

    def create_gendered_dataframes(self):
        """
        Creates one dataframe for "he" and one for "she".
        Each dataset consists of 173 pairs of sentences:
        each pair contains one gendered sentence and one that contains the occupation.
        """
        df = self.create_dataframe()

        # create men and women dataframes
        women = df[df['gender'] == 'woman']
        men = df[df['gender'] == 'man']

        # create copies of men and women dataframes
        men2 = women.copy()
        women2 = men.copy()

        # transform the copies to opposite gender ones
        if self.lang == 'sv':
            men2['sentence1'] = men2['sentence1'].apply(lambda x: self.replace_with(x, 'man'))
            women2['sentence1'] = women2['sentence1'].apply(lambda x: self.replace_with(x, 'kvinna'))
        if self.lang == 'en':
            men2['sentence1'] = men2['sentence1'].apply(lambda x: self.replace_with(x, 'man'))
            women2['sentence1'] = women2['sentence1'].apply(lambda x: self.replace_with(x, 'woman'))

        # concatenate dataframes of same gender
        women_df = pd.concat([women, women2])
        men_df = pd.concat([men, men2])

        # Uncomment next 2 lines to get  a "he"-and-"she" dataset instead of "a man"-and-"a woman" one
        women_df['sentence1'] = women_df['sentence1'].apply(lambda x: self.replace_with_he_she(x))
        men_df['sentence1'] = men_df['sentence1'].apply(lambda x: self.replace_with_he_she(x))

        # create t5-friendly or mT5-friendly inputs
        if self.multilinugal:
            women_df['input'] = "s1: " + women_df['sentence1'] + " s2: " + women_df['occupation']
            men_df['input'] = "s1: " + men_df['sentence1'] + " s2: " + men_df['occupation']
        else:
            women_df['input'] = "stsb sentence1: " + women_df['sentence1'] + " sentence2: " + women_df['occupation']
            men_df['input'] = "stsb sentence1: " + men_df['sentence1'] + " sentence2: " + men_df['occupation']

        # keep only inputs
        men_df = men_df[['sentence1', 'occupation', 'input']]
        men_df = men_df.sort_index(axis=0)
        women_df = women_df[['sentence1', 'occupation', 'input']]
        women_df = women_df.sort_index(axis=0)

        return women_df, men_df

    def create_dataframe(self):
        if self.lang == 'sv':
            test_path = os.path.join(self.data_dir, 'test-sv.tsv')
            df = pd.read_csv(test_path, sep='\t')
            print(f'len of test set: {df.shape}')
            df = df[['sentence1', 'sentence2']]
            # if sentence1 start with 'a man' or 'a woman' and lacks gender-pronouns, replace 'man' or 'woman'
            # with occupation in a new column:
            df['occupation'] = df['sentence1'].apply(
                lambda x: self.replace_with(x, self.occupation) if self.is_gendered(x) else False)
            # indicate the gender of a sentence in a new column :
            df['gender'] = df['sentence1'].apply(lambda x: 'man' if x[:6] == 'En man' else 'woman')
            # keep only those sentences which start with 'a man' or 'a woman' and lack gender-pronouns
            df = df[df.occupation != False]
            return df

        if self.lang == 'en':
            # infers on test set
            test_path = os.path.join(self.data_dir, 'sts-test.csv')
            df = pd.read_csv(test_path, delimiter='\t', header=None, quoting=csv.QUOTE_NONE, usecols=range(7))
            print(f'len of test set: {df.shape}')
            df = df.rename(columns={5: "sentence1", 6: "sentence2"})
            df = df[['sentence1', 'sentence2']]
            # if sentence1 start with 'en man' or 'en kvinna' and lacks gender-pronouns, replace 'man' or 'woman'
            # with occupation in a new column:
            df['occupation'] = df['sentence1'].apply(
                lambda x: self.replace_with(x, self.occupation) if self.is_gendered(x) else False)
            # indicate the gender of a sentence
            df['gender'] = df['sentence1'].apply(lambda x: 'man' if x[:5] == 'A man' else 'woman')
            # keep only those sentences which start with 'a man' or 'a woman' and lack gender-pronouns
            df = df[df.occupation != False]
            return df

    def is_gendered(self, x):
        if self.lang == 'sv':
            if x[:6] == 'En man':
                pronouns = ['hans', 'han', 'honom']
                if any([word in pronouns for word in [i.split('.')[0] for i in x.split(' ')]]):
                    return False
                else:
                    return True
            if x[:9] == 'En kvinna':
                pronouns = ['henne', 'hon', 'hennes']
                if any([word in pronouns for word in [i.split('.')[0] for i in x.split(' ')]]):
                    return False
                else:
                    return True

        if self.lang == 'en':
            if x[:5] == 'A man':
                pronouns = ['his', 'he', 'him']
                if any([word in pronouns for word in [i.split('.')[0] for i in x.split(' ')]]):
                    return False
                else:
                    return True
            if x[:7] == 'A woman':
                pronouns = ['her', 'she', 'hers']
                if any([word in pronouns for word in [i.split('.')[0] for i in x.split(' ')]]):
                    return False
                else:
                    return True

    def replace_with(self, x, occupation):
        if self.lang == 'sv':
            if x[:6] == 'En man':
                x = x.replace('En man ', 'En ' + occupation + ' ')
            if x[:9] == 'En kvinna':
                x = x.replace('En kvinna ', 'En ' + occupation + ' ')
            return x
        if self.lang == 'en':
            vowels = ['a', 'e', 'i', 'o', 'u']
            if x[:5] == 'A man':
                if occupation[0] in vowels:
                    x = x.replace('A man ', 'An ' + occupation + ' ')
                else:
                    x = x.replace('A man ', 'A ' + occupation + ' ')
            if x[:7] == 'A woman':
                if occupation[0] in vowels:
                    x = x.replace('A woman ', 'An ' + occupation + ' ')
                else:
                    x = x.replace('A woman ', 'A ' + occupation + ' ')
            return x

    def replace_with_he_she(self, x):
        if self.lang == 'sv':
            if x[:6] == 'En man':
                x = x.replace('En man ', 'Han ')
            if x[:9] == 'En kvinna':
                x = x.replace('En kvinna ', 'Hon ')
            return x
        if self.lang == 'en':
            if x[:5] == 'A man':
                x = x.replace('A man ', 'He ')
            if x[:7] == 'A woman':
                x = x.replace('A woman ', 'She ')
            return x

