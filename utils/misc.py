import os
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.inference import Inference
from dataloader.create_bias_dataset import CreateGenderStsb

careers = ['technician', 'accountant', 'supervisor', 'engineer', 'worker', 'educator', 'clerk', 'counselor',
           'inspector', 'mechanic', 'manager', 'therapist', 'administrator', 'salesperson', 'receptionist',
           'librarian', 'advisor', 'pharmacist', 'janitor', 'psychologist', 'physician', 'carpenter', 'nurse',
           'investigator', 'bartender', 'specialist', 'electrician', 'officer', 'pathologist', 'teacher', 'lawyer',
           'planner', 'practitioner', 'plumber', 'instructor', 'surgeon', 'veterinarian', 'paramedic', 'examiner',
           'chemist', 'machinist', 'appraiser', 'nutritionist', 'architect', 'hairdresser', 'baker', 'programmer',
           'paralegal', 'hygienist', 'scientist']

sv_careers = ['tekniker', 'revisor', 'handledare', 'ingenjör', 'arbetare', 'lärare', 'kontorist', 'rådgivare',
              'inspektör', 'mekaniker', 'chef', 'terapeut', 'administratör', 'försäljare', 'receptionist',
              'bibliotekarie', 'rådgivare', 'apotekare', 'vaktmästare', 'psykolog', 'läkare', 'snickare',
              'sjuksköterska', 'utredare', 'bartender', 'specialist', 'elektriker', 'officer', 'patolog', 'lärare',
              'advokat', 'planerare', 'utövare', 'rörmokare', 'instruktör', 'kirurg', 'veterinär', 'sjukvårdare',
              'granskare', 'kemist', 'maskinist', 'värderare', 'nutritionist', 'arkitekt', 'frisör', 'bagare',
              'programmerare', 'advokat', 'hygienist', 'forskare']



def clean_images_and_gifs():
    for filename in os.listdir('.'):
        if filename.endswith('.png') or filename.endswith('.gif'):
            os.remove(filename)


def create_all_occupations_df(hparams=None, multilingual=None):
    """
    Returns path to .csv of a dataframe that contains mean prediction values per gender for all occupations
    """
    all_occupations_df = pd.DataFrame(columns=['occupation', 'mean_women', 'mean_men'])
    model = Inference(hparams=hparams, multilingual=multilingual)
    lang = hparams['test_on']
    if lang == 'sv':
        occupations = sv_careers
    else:
        occupations = careers

    for i, occupation in enumerate(tqdm(occupations)):
        print(f'occupation {i + 1}/{len(careers)}...')
        dataset_creator = CreateGenderStsb(lang, data_dir=hparams['data_dir'], occupation=occupation,
                                           multilingual=multilingual)
        women_df, men_df = dataset_creator.create_gendered_dataframes()
        women_df['predictions'] = women_df['input'].apply(lambda x: float(model.predict(x)))
        men_df['predictions'] = men_df['input'].apply(lambda x: float(model.predict(x)))
        mean_w = women_df.predictions.mean()
        mean_m = men_df.predictions.mean()
        all_occupations_df.loc[i, 'occupation'] = occupation
        all_occupations_df.loc[i, 'mean_men'] = mean_m
        all_occupations_df.loc[i, 'mean_women'] = mean_w

    # create results folder
    results_path = 'results/bias_experiments'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # save final dataframe
    if multilingual:
        size = hparams['tokenizer_name_or_path'][11:]
        seed = hparams['seed']
        df_save_path = os.path.join(results_path, f'mt5_{size}_{lang}_{seed}.csv')
        all_occupations_df.to_csv(df_save_path, index=False)
    else:
        size = hparams['model_name_or_path'][3:]
        df_save_path = os.path.join(results_path, f't5_{size}.csv')
        all_occupations_df.to_csv(df_save_path, index=False)

    return df_save_path


def create_df_with_errors(all_dfs):
    """
    used in run_bias_experiments.py, only for mt5.
    all_dfs : a list of paths to .csvs for various seeds
    returns a final dataframe that includes statistical significance
    """
    size = str(all_dfs[0])[28:34].strip('_')
    lang = str(all_dfs[0])[34:37].strip('_')
    if len(all_dfs) == 1:  # if only one seed
        path = all_dfs[0]
        all_df = pd.read_csv(path)
    else:
        path1 = all_dfs[0]
        df1 = pd.read_csv(path1)
        for path in all_dfs[1:]:
            next_df = pd.read_csv(path)
            all_df = pd.concat(
                [df1[['mean_women', 'mean_men']], next_df[['mean_women', 'mean_men']]], axis=1)
            df1 = all_df

    means = all_df.groupby(by=all_df.columns, axis=1).mean()
    stds = all_df.groupby(by=all_df.columns, axis=1).std()
    stds.columns = ['man_std', 'woman_std']

    n = len(all_df.columns) / 2  # amount of seeds
    sem = stds.div(n)  # will be plotting the 'standard error of the mean' as the error margin
    sem2 = sem * 2  # we actually plot 2*sem = margin of the error,  which equals to confidence level of 95%
    sem2.columns = ['man_2*sem', 'woman_2*sem']

    occupations = pd.read_csv(all_dfs[0]).occupation
    final = pd.concat([occupations, means, sem2], axis=1)

    final = final.rename(index={key: value for (key, value) in zip(range(len(careers)), careers)})
    # save it
    results_path = 'results/bias_experiments'
    df_path: str = os.path.join(results_path, f'final_mt5_{size}_{lang}.csv')
    final.to_csv(df_path, index=False)

    return df_path


def plot_score_vs_occupation(path, multilingual=None):
    """
    used in run_bias_experiments.py
    path : path to final dataframe
    creates bar plot of all occupations and genders and saves.
    """
    if multilingual:
        results_path = 'results/bias_experiments'  # results/bias_experiments/final_mt5_small_en.csv
        df = pd.read_csv(path)
        lang = path[-6:-4]
        model = 'mT5'
        size = path[34:40].strip('_')
        df = df.rename(index={key: value for (key, value) in zip(range(len(careers)), careers)})
        df[['mean_men', 'mean_women']].plot(figsize=(20, 5), title=f'Model={model}\nSize={size}, Lang={lang}',
                                            kind='bar',
                                            yerr=df[['man_2*sem', 'woman_2*sem']].values.T,
                                            error_kw=dict(ecolor='k'),
                                            color=['royalblue', 'silver']
                                            )
        plt.tight_layout(pad=1)
        save_path = os.path.join(results_path, f'final_{lang}_{model}_{size}.png')
        plt.savefig(save_path)

    else:
        results_path = path[:24]
        size = path[28:33].strip('.')
        df = pd.read_csv(path)
        df[['mean_men', 'mean_women']].plot.bar(figsize=(20, 5), title=f'Model=T5 \nSize={size}',
                                                color=['royalblue', 'silver'])
        plt.xticks(range(len(df.occupation)), df.occupation)
        plt.tight_layout(pad=1)
        save_path = os.path.join(results_path, f't5_{size}.png')
        plt.savefig(save_path)


def plot_score_vs_size(paths: list, occupation: str, multilingual: bool):
    """
    :param t5_dfs_paths: list of paths to final dataframes' csvs.
    It can include be multiple files per size in case of mt5
    :param occupation: the selected occupation to be ploted
    saves a .png (in case of mT5, saves one per language)
    """
    if multilingual:
        en_paths = []
        sv_paths = []
        en_df1 = pd.DataFrame(columns=['mean_men', 'mean_women', 'man_2*sem', 'woman_2*sem'])
        sv_df1 = pd.DataFrame(columns=['mean_men', 'mean_women', 'man_2*sem', 'woman_2*sem'])
        for path in paths:
            lang = path[-6:-4]
            if lang == 'en':
                en_paths.append(path)
            if lang == 'sv':
                sv_paths.append(path)
        for path in en_paths:  # results/bias_experiments/final_mt5_small_en.csv
            size = path[34:40].strip('_')
            next_df = pd.read_csv(path)
            next_df = next_df[next_df.occupation == occupation]
            next_df['size'] = size
            df = pd.concat([en_df1, next_df], axis=0)
            en_df1 = df
        for path in en_paths:
            size = path[34:40].strip('_')
            next_df = pd.read_csv(path)
            next_df = next_df[next_df.occupation == occupation]
            next_df['size'] = size
            df = pd.concat([sv_df1, next_df], axis=0)
            sv_df1 = df
        # save english graph
        en_df1[['mean_men', 'mean_women']].plot.bar(figsize=(20, 5),
                                                    title=f'Occupation: {occupation} vs. mT5 model sizes\nLang=en',
                                                    yerr=en_df1[['man_2*sem', 'woman_2*sem']].values.T,
                                                    error_kw=dict(ecolor='k'),
                                                    color=['royalblue', 'silver'])
        plt.xticks(range(len(en_df1['size'])), en_df1['size'])
        save_path = os.path.join('results/bias_experiments', f'mt5_{occupation}_vs_sizes_en.png')
        plt.savefig(save_path)
        # save swedish graph
        sv_df1[['mean_men', 'mean_women']].plot.bar(figsize=(20, 5),
                                                    title=f'Occupation: {occupation} vs. mT5 model sizes\nLang=en',
                                                    yerr=sv_df1[['man_2*sem', 'woman_2*sem']].values.T,
                                                    error_kw=dict(ecolor='k'),
                                                    color=['royalblue', 'silver'])
        plt.xticks(range(len(sv_df1['size'])), sv_df1['size'])
        save_path = os.path.join('results/bias_experiments', f'mT5_{occupation}_vs_sizes_sv.png')
        plt.savefig(save_path)
    else:
        df1 = pd.DataFrame(columns=['occupation', 'mean_women', 'mean_men', 'size'])
        for path in paths:  # 'results/bias_experiments/t5_small.csv'
            size = path[28:33].strip('.')
            next_df = pd.read_csv(path)
            next_df = next_df[next_df.occupation == occupation]
            next_df['size'] = size
            df = pd.concat([df1, next_df], axis=0)
            df1 = df
        df1[['mean_men', 'mean_women']].plot.bar(figsize=(20, 5), title=f'Occupation: {occupation} vs. T5 model sizes',
                                                 color=['royalblue', 'silver'])
        plt.xticks(range(len(df1['size'])), df1['size'])
        save_path = os.path.join('results/bias_experiments', f'T5_{occupation}_vs_sizes.png')
        plt.savefig(save_path)


########### croos ling funcs ############

def create_final_df(path, bucket, size):
    '''
    used by run_stsb_crossling_experiments.py
    :param path: receives a path to a temporary dataframe that has holds the results for different restarts of the four
    cross-lingual experiments. (1: en->en, 2: en->sv, 3: sv->en, 4: sv->sv)
    :return: a final dataframe that provides the mean pcc per language setting alongside with the corresponding errors
    '''
    df = pd.read_csv(path)
    df['train-test language'] = 'train_on: ' + df['train_on'] + '  ' + 'test_on: ' + df['test_on']
    df.drop(['train_on', 'test_on'], axis=1, inplace=True)
    mean_pcc = df.groupby('train-test language')['pcc'].mean()  # mean pcc over different restarts(seeds)
    std = df.groupby('train-test language')['pcc'].std()  # std
    n = df.groupby('train-test language')['seed'].count()  # number of restarts
    sem = std.div(n)  # will be plotting the 'standard error of the mean' as the error margin
    sem2 = sem * 2  # we actually plot 2*sem = margin of the error,  which equals to confidence level of 95%
    # (or statistical significance of 5%). we do that to ensure a meaningful comparison over different pcc-means
    # concise justification of this choice can be found here:
    # https://nyuwinthrop.orgtheory/wp-content/uploads/2019/08/standard-deviation-standard-error.pdf

    final_df = pd.concat([mean_pcc, sem2], axis=1)
    final_df.columns = ['mean', '2*sem']
    # remove the temporary csv
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    # create results folder
    save_path = 'results/crosslingual_experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_path: str = os.path.join(save_path, f'mt5_{size}_bucket_{bucket}.csv')
    final_df.to_csv(df_path, index=False)
    return final_df


def plot_cross_lingual(final_df, bucket, size):
    '''
    :param final_df: receives a final dataframe with mean pcc ± error for all 4 cross-lingual settings
    :param bucket: bucket mode used in quantization of the similarity score.
    :return: nothing. saves a png of the graph in directory: results/crosslingual_experiments
    '''
    my_colors = [(x / 4.0, x / 10.0, 0.75) for x in
                 range(len(final_df))]  # <-- Quick gradient example along the Red/Green dimensions.
    ax = final_df.plot(kind="barh", y="mean",
                       title=f"Pearson Correlation Coefficient vs. Language Setting\n\nBucket={bucket}\n\nSize={size}\n",
                       xerr="2*sem",
                       legend=None,
                       color=my_colors)
    ax.set_xlabel("Mean pcc over different seeds ± 2*SEM")
    ax.set_ylabel("Language")
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))
    plt.tight_layout(pad=1)
    save_path = 'results/crosslingual_experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    png_path: str = os.path.join(save_path, f'mt5_{size}_bucket_{bucket}.png')
    fig = ax.get_figure()
    fig.savefig(png_path)

#################################
