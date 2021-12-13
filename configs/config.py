import os
import torch
import argparse

if not torch.cuda.is_available():
    os.environ['WANDB_SILENT'] = 'true'
    os.environ["WANDB_MODE"] = 'dryrun'
    os.environ['OMP_NUM_THREADS'] = '1'
    n_gpu = 0
else:
    n_gpu = 1


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', help='directory that holds the dataset')
    parser.add_argument('-o', '--output_dir', help='directory to save fine-tuned model and checkpoints')
    parser.add_argument('-m', '--model_name_or_path', default='google/mt5-base', help='path to model')
    parser.add_argument('-k', '--tokenizer_name_or_path', default='google/mt5-base', help='path to tokenizer')
    parser.add_argument('-l', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-g', '--n_gpu', default=n_gpu)
    parser.add_argument('-s', '--seed', default=42)
    parser.add_argument('-b', '-train_batch_size', '--train_batch_size', default=16, type=int)
    parser.add_argument('-v', '-eval_batch_size', '--eval_batch_size', default=16, type=int)
    parser.add_argument('-e', '-num_train_epochs', '--num_train_epochs', default=3, type=int)
    parser.add_argument('-d', '-debug', '--debug', default=False, help='sets to debug mode')
    parser.add_argument('-t', '-train_on', '--train_on', default='en',
                        help='choose language to train on: either "en" or "sv"')
    parser.add_argument('-z', '-test_on', '--test_on', default='en',
                        help='choose language to test on: either "en" or "sv"')
    parser.add_argument('-w', '--wandb_proj_name', default='STSb-EN-final',
                        help='define weights and biases project name')
    parser.add_argument('--max_seq_length', default=256, type=int)
    parser.add_argument('--max_target_len', default=2, type=int)
    parser.add_argument('--debias', default=False)
    parser.add_argument('--bucket_mode', default=1.0, type=float,
                        help='quantization of similarity score. can be 1.0, 0.2 or 0.1')
    parser.add_argument('--adam_epsilon', default=1e-4, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--dropout_rate', default=0.007, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_steps', '--warmup_steps', default=0, type=int)
    parser.add_argument('--num_sanity_val_steps', default=0, type=int)
    parser.add_argument('--mse_loss', default=False)
    parser.add_argument('--normalize', default=True)
    parser.add_argument("--occupation', default='technician', help='choose occupation to plot score-vs-model-sizes for. Available occupations: ['technician', 'accountant', 'supervisor', 'engineer', 'worker', 'educator', 'clerk', 'counselor',
           'inspector', 'mechanic', 'manager', 'therapist', 'administrator', 'salesperson', 'receptionist',
           'librarian', 'advisor', 'pharmacist', 'janitor', 'psychologist', 'physician', 'carpenter', 'nurse',
           'investigator', 'bartender', 'specialist', 'electrician', 'officer', 'pathologist', 'teacher', 'lawyer',
           'planner', 'practitioner', 'plumber', 'instructor', 'surgeon', 'veterinarian', 'paramedic', 'examiner',
           'chemist', 'machinist', 'appraiser', 'nutritionist', 'architect', 'hairdresser', 'baker', 'programmer',
           'paralegal', 'hygienist', 'scientist']")
    return parser
