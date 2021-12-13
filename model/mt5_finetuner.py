import emoji as em
import wandb
import pytorch_lightning as pl
from dataloader.dataloader import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from transformers import AdamW, MT5ForConditionalGeneration, T5Tokenizer
from evaluation.classification_task_eval import *
from utils.misc import *
from configs.config import *

if not torch.cuda.is_available():
    os.environ['WANDB_SILENT'] = 'true'
    os.environ["WANDB_MODE"] = 'dryrun'
    os.environ['OMP_NUM_THREADS'] = '1'

class MT5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(MT5FineTuner, self).__init__()
        self.hparams = argparse.Namespace(**hparams)
        # print(f'\nSpecified parameters: \n{self.hparams}')
        self.model = MT5ForConditionalGeneration.from_pretrained(
            hparams.model_name_or_path, dropout_rate=self.hparams.dropout_rate)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.num_classes = int((5 / self.hparams.bucket_mode) + 1)  # TODO

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):
        """
        Args:
            input_ids: (torch.Tensor of shape (batch_size, max_seq_length)) - The tokenized inputs.
            attention_mask: (torch.Tensor of shape (batch_size, max_seq_length)) The input attention masks.
            decoder_attention_mask: (torch.Tensor of shape (batch_size, max_target_len)) The target attention masks.
            labels: (torch.Tensor of shape (batch_size, max_target_len)) The tokenized targets.

        Returns:
             A Seq2SeqLMOutput comprising : ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: dict comprising of ['source_ids', 'source_mask', 'target_ids', 'target_mask', 'target']

        Returns:
            MT5ForConditionalGeneration Language modelling loss
        """
        # Ensures pad tokens are set to -100. Loss is only computed for labels in [0, ..., config.vocab_size].
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        self.logger.experiment.log({'batch_train_loss': loss, 'epoch': self.current_epoch, 'step': self.global_step})

        return loss

    def training_epoch_end(self, outputs):
        """
        Args:
            outputs: list of length (num_samples // batch_size) comprising of the loss of each batch
        Returns:
            Nothing. Logs mean loss per epoch.
        """
        avg_loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.logger.experiment.log({'avg_train_loss': avg_loss, 'epoch': self.current_epoch, 'step': self.global_step})

    def validation_step(self, batch, batch_idx):
        """
            Args:
                batch: dict comprising of ['source_ids', 'source_mask', 'target_ids', 'target_mask', 'target']
            Returns:
                dict comprising of LM loss, batch-targets and batch-predictions
        """
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )
        loss = outputs[0]
        generated = self.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"]).squeeze()

        predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
        targetids = lm_labels
        targetids[targetids[:, :] == -100] = 0
        targets = self.tokenizer.batch_decode(targetids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return {'val_loss': loss, 'targets': targets, 'predictions': predictions}

    def validation_epoch_end(self, outputs):
        """
        Args:
             outputs: list of length (num_samples // val_batch_size) containing losses of each batch
        Returns:
            Nothing. Logs mean validation loss and pcc per validation epoch. Saves confusion matrix per epoch.
        """

        val_loss = torch.stack([output["val_loss"] for output in outputs]).mean()
        predictions = []
        labels = []
        for output in outputs:
            predictions.extend(output['predictions'])
            labels.extend(output['targets'])
        predictions = [torch.Tensor([float(i)]) for i in predictions]
        labels = [torch.Tensor([float(i)]) for i in labels]

        metrics_dict = classification_metrics(predictions, labels)
        save_conf_mat(predictions, labels, current_epoch=self.current_epoch, current_run=wandb.run, mode='eval',
                      bucket_mode=self.hparams.bucket_mode)

        self.logger.experiment.log({'val_loss': val_loss,
                                    'validation_pearson_cor_coef': metrics_dict['pearson_cor_coef'],
                                    'epoch': self.current_epoch,
                                    'step': self.global_step})

    def test_step(self, batch, batch_idx):

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )
        loss = outputs[0]

        generated = self.model.generate(input_ids=batch["source_ids"],
                                        attention_mask=batch["source_mask"]).squeeze()

        predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
        targetids = lm_labels
        targetids[targetids[:, :] == -100] = 0
        targets = self.tokenizer.batch_decode(targetids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return {'test_loss': loss, 'targets': targets, 'predictions': predictions}

    def test_epoch_end(self, outputs):

        # save eval gif
        prepare_gif()
        self.logger.experiment.log(
            {"Validation Set": wandb.Video('./out.gif', fps=4, format="gif"), 'step': self.global_step})
        clean_images_and_gifs()

        test_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        predictions = []
        labels = []
        for output in outputs:
            predictions.extend(output['predictions'])
            labels.extend(output['targets'])
        predictions = [torch.Tensor([float(i)]) for i in predictions]
        labels = [torch.Tensor([float(i)]) for i in labels]

        save_conf_mat(predictions, labels, current_epoch=self.current_epoch, current_run=wandb.run, mode='test',
                      bucket_mode=self.hparams.bucket_mode)
        self.logger.experiment.log(
            {"Test Set": wandb.Video('./test.gif', fps=4, format="gif"), 'step': self.global_step})
        os.remove('test_confusion_matrix.png')
        os.remove('test.gif')
        metrics_dict = classification_metrics(predictions, labels)
        self.logger.experiment.log({'test_loss': torch.Tensor([test_loss]), **metrics_dict, 'step': self.global_step})
        pcc = metrics_dict['pearson_cor_coef']
        self.pcc = pcc
        return pcc


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [

            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure=None, using_native_amp=False,
                       on_tpu=False, using_lbfgs=False):
        super(MT5FineTuner, self).optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            using_native_amp=using_native_amp,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_lbfgs=using_lbfgs)
        optimizer.step()
        optimizer.zero_grad()
        # self.lr_scheduler.step()

    def train_dataloader(self):
        if self.hparams.train_on == 'en':
            train_dataset = EnDataset(type_path="sts-train", hparams=self.hparams)
        elif self.hparams.train_on == 'sv':
            train_dataset = SvDataset(type_path="train-sv", hparams=self.hparams)
        elif self.hparams.train_on == 'mix':
            train_dataset = MixedDataset(hparams=self.hparams, en_type_path="sts-train", sv_type_path="train-sv")
        else:
            raise ValueError('Unknown language: choose "en","sv" or "mix"')
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        '''
        total_num_steps = ((len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, int(
            self.hparams.n_gpu)))) // float(self.hparams.num_train_epochs)) * self.hparams.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=total_num_steps)
        self.lr_scheduler = scheduler
        '''
        return dataloader

    def val_dataloader(self):
        if self.hparams.train_on == 'en':
            val_dataset = EnDataset(type_path="sts-dev", hparams=self.hparams)
        elif self.hparams.train_on == 'sv':
            val_dataset = SvDataset(type_path="dev-sv", hparams=self.hparams)
        elif self.hparams.train_on == 'mix':
            val_dataset = MixedDataset(hparams=self.hparams, en_type_path="sts-dev", sv_type_path="dev-sv")
        else:
            raise ValueError('Unknown language: choose "en","sv" or "mix"')
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

    def test_dataloader(self):
        if self.hparams.test_on == 'en':
            test_dataset = EnDataset(type_path="sts-test", hparams=self.hparams)
        elif self.hparams.test_on == 'sv':
            test_dataset = SvDataset(type_path="test-sv", hparams=self.hparams)
        else:
            raise ValueError('Unknown language: choose "en" or "sv"')
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


def train(hp_defaults: dict, save_model=False, test_model=False):

    wandb.init(config=hp_defaults, project='STSb-EN-final')

    config = wandb.config
    wandb.config.update(hp_defaults, allow_val_change=True)
    print(f'\nSpecified hyperparameters for current experiment:\n\n{config}\n\n')

    pl.seed_everything(config.seed)
    wandb_logger = WandbLogger()
    run_name = wandb.wandb.run.name or 'no_name'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, run_name),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename='{epoch}-{val_loss:.2f}')

    train_params = dict(
        # accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=config.n_gpu,
        max_epochs=config.num_train_epochs,
        gradient_clip_val=config.max_grad_norm,
        logger=wandb_logger,
        default_root_dir=config.output_dir,
        num_sanity_val_steps=config.num_sanity_val_steps,
        # checkpoint_callback=checkpoint_callback if save_model else False
    )

    model = MT5FineTuner(config)
    trainer = pl.Trainer(**train_params)
    print(em.emojize('\n:sparkles: Starts Training :sparkles:\n'))
    trainer.fit(model)
    print(em.emojize(f'\nTraining is done :check_mark:\n'))

    if test_model:
        trainer.test(model)
        pcc = model.pcc
        return pcc
    if save_model:
        model.model.save_pretrained(os.path.join(config.output_dir, run_name))
        return str(os.path.join(config.output_dir, run_name))
