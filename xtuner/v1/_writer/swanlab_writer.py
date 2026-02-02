import os
from pathlib import Path

import swanlab
from mmengine.dist import get_rank
from swanlab.plugin.notification import LarkCallback

SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
SWANLAB_WORKSPACE = os.environ.get("SWANLAB_WORKSPACE", None)
SWANLAB_PROJECT = os.environ.get("SWANLAB_PROJECT", None)
SWANLAB_EXPERIMENT = os.environ.get("SWANLAB_EXPERIMENT", None)
SWANLAB_EXP_DESCRIPTION = os.environ.get("SWANLAB_EXP_DESCRIPTION", None)
SWANLAB_EXP_GROUP = os.environ.get("SWANLAB_EXP_GROUP", None)
SWANLAB_LARK_WEBHOOK_URL = os.environ.get("SWANLAB_LARK_WEBHOOK_URL", None)
SWANLAB_LARK_WEBHOOK_SECRET = os.environ.get("SWANLAB_LARK_WEBHOOK_SECRET", None)


class SwanlabWriter:
    def __init__(
        self,
        log_dir: str | Path | None = None,
        **kwargs,
    ):
        if log_dir is None:
            log_dir = Path()

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        # Only init swanlab on rank 0
        self._rank = get_rank()
        if self._rank == 0:
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)
            if SWANLAB_LARK_WEBHOOK_URL and SWANLAB_LARK_WEBHOOK_SECRET:
                lark_callback = LarkCallback(
                    webhook_url=SWANLAB_LARK_WEBHOOK_URL,  # type: ignore
                    secret=SWANLAB_LARK_WEBHOOK_SECRET,  # type: ignore
                )
            else:
                lark_callback = None

            config = kwargs.get("config")
            if config is not None:
                trainer_config = config.model_dump(mode="json")
            else:
                trainer_config = None

            swanlab.init(
                workspace=SWANLAB_WORKSPACE or None,  # type: ignore
                project=SWANLAB_PROJECT or None,  # type: ignore
                experiment_name=SWANLAB_EXPERIMENT or None,  # type: ignore
                description=SWANLAB_EXP_DESCRIPTION or None,  # type: ignore
                group=SWANLAB_EXP_GROUP or None,  # type: ignore
                logdir=SWANLAB_LOG_DIR or log_dir,
                mode=SWANLAB_MODE,  # type: ignore
                config=trainer_config,
                callbacks=[lark_callback] if lark_callback else None,
            )
            self._writer = swanlab
        else:
            self._writer = None

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        # Only log on rank 0
        if self._rank == 0 and self._writer is not None:
            self._writer.log({tag: scalar_value}, step=global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        # Only log on rank 0
        if self._rank == 0 and self._writer is not None:
            self._writer.log(tag_scalar_dict, step=global_step)

    def close(self):
        if self._rank == 0 and self._writer is not None:
            self._writer.finish()
