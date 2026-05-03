import os
import traceback

import torch
from lightning import Trainer
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from rich import traceback as rich_traceback

from src import dataset as datasets
from src.config import CheckpointMonitor, Config
from src.model.base import BaseDeepakeDetectionModel
from src.utils import logger
from src.utils.checks import checks
from src.utils.model_checkpoint import ModelCheckpointParallel

rich_traceback.install()


def load_third_party_model(config: Config) -> BaseDeepakeDetectionModel:
    if "weights/Effort" in config.checkpoint:
        # Download: https://drive.google.com/drive/folders/19kQwGDjF18uk78EnnypxxOLaG4Aa4v1h
        from src.model.Effort import Effort

        return Effort(config)

    if "weights/ForAda" in config.checkpoint:
        # Download: https://drive.usercontent.google.com/download?id=1UlaAUTtsX87ofIibf38TtfAKIsnA7WVm&export=download&authuser=0
        from src.model.ForAda import ForAda

        return ForAda(config)

    if "weights/FS-VFM/" in config.checkpoint:
        from src.model.FSFM import FSFM

        return FSFM(config)

    if "yermandy/" in config.checkpoint:
        # https://huggingface.co/yermandy/models
        from src.model.GenDHF import GenDHF

        return GenDHF(config)


    raise ValueError(f"Unknown third party model in checkpoint path: {config.checkpoint}")


def load_model(config: Config) -> BaseDeepakeDetectionModel:
    # If no checkpoint is provided, use GenD as default
    if config.checkpoint is None or config.checkpoint == "":
        from src.model.GenD import GenD

        return GenD(config, verbose=True)

    # Try to load third party model
    try:
        return load_third_party_model(config)
    except ValueError:
        # If not a third party model, use GenD as default
        from src.model.GenD import GenD

        return GenD(config, verbose=True)


def init_loggers(config: Config) -> list:
    save_dir = f"{config.run_dir}/{config.run_name}"

    loggers: list = [pl_loggers.CSVLogger(config.run_dir, name=config.run_name, version="")]

    if config.wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project="deepfake",
            name=config.wandb_name or config.run_name,
            save_dir=save_dir,
            tags=set(config.wandb_tags),
            group=config.wandb_group,
        )
        loggers.append(wandb_logger)

    return loggers


def get_checkpoint_monitors(config: Config) -> list[CheckpointMonitor]:
    if config.checkpoint_monitors:
        return config.checkpoint_monitors
    return [
        CheckpointMonitor(
            filename=config.checkpoint_name,
            monitor=config.monitor_metric,
            mode=config.monitor_metric_mode,
        )
    ]


def init_callbacks(config: Config, enable_checkpointing: bool = True) -> list:
    callbacks = [pl_callbacks.RichProgressBar(leave=True)]
    if enable_checkpointing:
        callbacks.extend(
            ModelCheckpointParallel(filename=monitor.filename, monitor=monitor.monitor, mode=monitor.mode)
            for monitor in get_checkpoint_monitors(config)
        )
    # pl_callbacks.LearningRateFinder(1e-5, 1e-2),

    if config.early_stopping_patience > 0:
        callbacks.append(
            pl_callbacks.EarlyStopping(
                monitor=config.monitor_metric,
                patience=config.early_stopping_patience,
                mode=config.monitor_metric_mode,
                verbose=True,
            )
        )

    return callbacks


def init_trainer(config: Config, enable_checkpointing: bool) -> Trainer:
    return Trainer(
        devices=config.devices,
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=config.batch_size // config.mini_batch_size,
        fast_dev_run=config.fast_dev_run,
        overfit_batches=config.overfit_batches,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        deterministic=config.deterministic,
        detect_anomaly=config.detect_anomaly,
        logger=init_loggers(config),
        callbacks=init_callbacks(config, enable_checkpointing),
        default_root_dir=config.run_dir,
        log_every_n_steps=100,
    )


def finish_wandb_run(trainer, config: Config):
    if config.wandb:
        if any(isinstance(l, pl_loggers.WandbLogger) for l in trainer.loggers):
            wandb_logger = [l for l in trainer.loggers if isinstance(l, pl_loggers.WandbLogger)][0]
            wandb_logger.finalize("success")
            wandb_logger.experiment.finish()


def get_post_train_test_checkpoint_names(config: Config) -> list[str]:
    if config.post_train_test_checkpoint_names:
        return config.post_train_test_checkpoint_names
    return [monitor.filename for monitor in get_checkpoint_monitors(config)]


def run_checkpoint_test(config: Config, checkpoint_name: str, checkpoint_path: str):
    test_config = Config(**config.model_dump())
    test_config.checkpoint = checkpoint_path
    test_config.run_dir = f"{config.run_dir}/{config.run_name}/checkpoint_tests"
    test_config.run_name = checkpoint_name
    test_config.wandb_name = f"{config.run_name}-test-{checkpoint_name}" if config.wandb else None
    test_config.throw_exception_if_run_exists = False
    test_config.remove_if_run_exists = True

    checks(test_config)
    torch.set_float32_matmul_precision("high")

    model = load_model(test_config)
    model.load_checkpoint(test_config.checkpoint)
    data_module = datasets.DeepfakeDataModule(test_config, model.get_preprocessing())
    trainer = init_trainer(test_config, enable_checkpointing=False)
    trainer.test(model, data_module)
    finish_wandb_run(trainer, test_config)


def main(config: Config, train: bool):
    # Performs initial checks
    checks(config)

    # Set the precision for matmul operations
    torch.set_float32_matmul_precision("high")

    # Instantiates the model
    model = load_model(config)

    # Loads the checkpoint if provided
    model.load_checkpoint(config.checkpoint)

    data_module = datasets.DeepfakeDataModule(config, model.get_preprocessing())

    save_dir = f"{config.run_dir}/{config.run_name}"

    trainer = init_trainer(config, enable_checkpointing=train)

    if train:
        try:
            trainer.fit(model, data_module)
        except KeyboardInterrupt:
            logger.print_warning("Training interrupted")
        except Exception as e:
            traceback.print_exc()  # Print complete exception traceback
            logger.print_error(f"Training failed: {e}")
            # Save the exception traceback to a file
            with open(f"{save_dir}/failed.log", "a") as f:
                f.write(f"Training failed: {e}\n")
                f.write(traceback.format_exc())
        finally:
            if config.checkpoint_monitors or config.post_train_test_checkpoint_names:
                finish_wandb_run(trainer, config)
                logger.print_info("Training finished. Starting checkpoint testing")
                del trainer, model, data_module
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for checkpoint_name in get_post_train_test_checkpoint_names(config):
                    ckpt_path = f"{save_dir}/checkpoints/{checkpoint_name}.ckpt"
                    if not os.path.exists(ckpt_path):
                        logger.print_error(f"Checkpoint {ckpt_path} does not exist. Cannot proceed with testing.")
                        continue
                    run_checkpoint_test(config, checkpoint_name, ckpt_path)
            else:
                logger.print_info("Training finished. Starting testing")
                ckpt_path = f"{save_dir}/checkpoints/{config.checkpoint_name}.ckpt"
                if not os.path.exists(ckpt_path):
                    logger.print_error(f"Checkpoint {ckpt_path} does not exist. Cannot proceed with testing.")
                else:
                    model.load_checkpoint(ckpt_path)
                    trainer.test(model, data_module)
                finish_wandb_run(trainer, config)

    else:
        assert config.checkpoint is not None, "Checkpoint is required for testing"
        trainer.test(model, data_module)
        finish_wandb_run(trainer, config)
