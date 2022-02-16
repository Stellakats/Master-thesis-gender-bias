import torch
from utils.misc import *
from model import mt5_finetuner
import configs.config as config

# experiment-specific args are defined here

hyperparameter_defaults = dict(
    data_dir='./data/stsbenchmark/',
    output_dir='./checkpoints_stsb-en',
    n_gpu=1 if torch.cuda.is_available() else 0,
    model_name_or_path='google/mt5-small',
    tokenizer_name_or_path='google/mt5-small',
    max_seq_length=256,
    max_target_len=2,
    learning_rate=1e-4,
    adam_epsilon=1e-4,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    seed=42,
    dropout_rate=0.007,
    weight_decay=0.05,
    warmup_steps=0,
    bucket_mode=1.0,
    num_sanity_val_steps=0,
    train_batch_size=16,
    eval_batch_size=16,
    num_train_epochs=3,
    debug=False,
    mse_loss=False,
    normalize=True,
    train_on='en',  # ['en','sv','mix']
    test_on='en',
    debias=False,
    occupation='technician'
)


def main():
    # Experiments to detect bias on different models (T5 and mT5), languages and sizes.
    # For T5, the experiments include inference only.
    # For mT5, the model is first fine-tuned on the task.

    # fetch and update parser; command line args are prioritized if given
    parser = config.build_parser()
    parser.set_defaults(**hyperparameter_defaults)
    args = vars(parser.parse_args())

    t5_sizes = ['t5-small', 't5-base', 't5-large']
    mt5_sizes = ['google/mt5-small', 'google/mt5-base', 'google/mt5-large']
    languages = ['en', 'sv']
    seeds = [42, 43, 44]
    total_exp_count = len(mt5_sizes) * len(languages) * len(seeds) + len(t5_sizes)
    exp_count = 0

    # T5:

    t5_dfs_paths = []
    for size in t5_sizes:
        exp_count += 1
        print(f'{100 * "*"}\nRunning experiment {exp_count}/{total_exp_count}...\n{100 * "*"}\n')
        # update arg-dictionary
        hyperparameter_defaults['train_on'] = 'en'
        hyperparameter_defaults['test_on'] = 'en'
        hyperparameter_defaults['model_name_or_path'] = size
        hyperparameter_defaults['tokenizer_name_or_path'] = size
        # update parser
        parser.set_defaults(**hyperparameter_defaults)
        args = vars(parser.parse_args())
        print(args)
        # creates a new gender-stsb dataset, infers on it using the above fine-tuned model
        # and finally saves a dataframe for all occupations
        final_df_path = create_all_occupations_df(hparams=args, multilingual=False)
        t5_dfs_paths.append(final_df_path)
        # plot final dataframe
        plot_score_vs_occupation(final_df_path, multilingual=False)
    # plot score vs sizes for a particular occupation 
    plot_score_vs_size(paths=t5_dfs_paths, occupation=args['occupation'], multilingual=False)

    # mT5:

    mt5_final_dfs_paths = []
    for language in languages:
        for size in mt5_sizes:
            mt5_dfs_paths = []
            for seed in seeds:
                exp_count += 1
                print(f'{100 * "*"}\nRunning experiment {exp_count}/{total_exp_count} (size={size[11:]}, '
                      f'lang={language}, seed={seed})...\n{100 * "*"}\n')
                # update arg-dictionary
                hyperparameter_defaults['train_on'] = language
                hyperparameter_defaults['test_on'] = language
                hyperparameter_defaults['model_name_or_path'] = size
                hyperparameter_defaults['tokenizer_name_or_path'] = size
                hyperparameter_defaults['seed'] = seed
                # update parser
                parser.set_defaults(**hyperparameter_defaults)
                args = vars(parser.parse_args())
                # train and save
                finetuned_model = mt5_finetuner.train(hp_defaults=args, save_model=True,
                                                      test_model=False)  # returns the path of saved model
                # update arg-dictionary on the new model
                hyperparameter_defaults['model_name_or_path'] = finetuned_model
                parser.set_defaults(**hyperparameter_defaults)
                args = vars(parser.parse_args())
                # creates a new gender-stsb dataset, infers on it using the above fine-tuned model
                # and finally saves a dataframe for all occupations
                df_path = create_all_occupations_df(hparams=args, multilingual=True)  # returns path
                # collect dfs for all seeds of the same experiment in a list
                mt5_dfs_paths.append(df_path)
            
            #  make a final df with error bars
            final_df_path = create_df_with_errors(mt5_dfs_paths)
            mt5_final_dfs_paths.append(final_df_path)
            print(f'{100 * "*"}\nFinished all experiments for {size[11:]} size of mT5 - lang={language}).\n'
                  f'Saving final graph...\n{100 * "*"}\n')
            # plot final df
            plot_score_vs_occupation(final_df_path, multilingual=True)

    # plot score vs sizes for a particular occupation (saves one png per language)
    mt5_final_dfs_paths = ['results/bias_experiments/final_mt5_small_en.csv',
                           'results/bias_experiments/final_mt5_small_sv.csv',
                           'results/bias_experiments/final_mt5_base_en.csv',
                           'results/bias_experiments/final_mt5_base_sv.csv',
                           'results/bias_experiments/final_mt5_large_en.csv',
                           'results/bias_experiments/final_mt5_large_sv.csv'
                           ]
    plot_score_vs_size(paths=mt5_final_dfs_paths, occupation=args['occupation'], multilingual=True)


if __name__ == '__main__':
    main()
