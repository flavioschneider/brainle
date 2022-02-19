import os
import dotenv
import hydra
import pytorch_lightning as pl
import logging
from hydra.utils import call, instantiate
from omegaconf import DictConfig

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)
log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    pl.seed_everything(config.seed)

    if config.type == "script":
        script = instantiate(config.script)
        script.run()

    elif config.type == "model":

        # Initialize datamodule
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
        datamodule = instantiate(config.datamodule)

        # Initialize model
        log.info(f"Instantiating model <{config.model._target_}>.")
        model = instantiate(config.model)

        # Initialize all callbacks (e.g. checkpoints, early stopping)
        callbacks = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>.")
                    callbacks.append(instantiate(cb_conf))

        # Initialize loggers (e.g. comet-ml)
        loggers = []
        if "loggers" in config:
            for _, lg_conf in config["loggers"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>.")
                    loggers.append(instantiate(lg_conf))

        # Initialize trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>.")
        trainer = instantiate(
            config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
        )

        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
