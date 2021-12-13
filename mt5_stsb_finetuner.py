import torch
import emoji as em
from utils.misc import *
from model import mt5_finetuner
import configs.config as config

# set experiment-specific args:
hyperparameter_defaults = dict(
    data_dir='./data/stsbenchmark/',
    output_dir='./ckeckpoints_stsb',
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
    dropout_rate=0.01,
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
    debias=False
)


def main():
    # fetch and update parser; command line args are prioritized if given
    parser = config.build_parser()
    parser.set_defaults(**hyperparameter_defaults)
    args = vars(parser.parse_args())
    # train
    pcc = mt5_finetuner.train(hp_defaults=args, save_model=True, test_model=True)
    print(f'PCC on test set:{pcc}')

if __name__ == '__main__':
    main()
