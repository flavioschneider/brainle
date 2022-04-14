import logging
import os

import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from brainle import utils

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)
log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    # Logs config tree
    # utils.extras(config)

    # Apply seed for reproducibility
    pl.seed_everything(config.seed)

    if config.type == "script":
        script = hydra.utils.instantiate(config.script)
        script.run()

    elif config.type == "model":

        # Initialize datamodule
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
        datamodule = hydra.utils.instantiate(config.datamodule)

        # Initialize model
        log.info(f"Instantiating model <{config.model._target_}>.")
        model = hydra.utils.instantiate(config.model)

        # Initialize all callbacks (e.g. checkpoints, early stopping)
        callbacks = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>.")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Initialize loggers (e.g. comet-ml)
        loggers = []
        if "loggers" in config:
            for _, lg_conf in config["loggers"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>.")
                    loggers.append(hydra.utils.instantiate(lg_conf))

        # Initialize trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>.")
        trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
        )

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=loggers,
        )

        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)

        # Test the model
        if config.get("test"):
            ckpt_path = "best"
            if not config.get("train") or config.trainer.get("fast_dev_run"):
                ckpt_path = None
            log.info("Starting testing!")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=loggers,
        )

        # Print path to best checkpoint
        if not config.trainer.get("fast_dev_run") and config.get("train"):
            log.info(
                f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}"
            )


if __name__ == "__main__":
    main()
