from typing import Dict, Optional
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger as _pl_WandbLogger
import os
from pathlib import Path


# Fix the step logging
class WandbLogger(_pl_WandbLogger):
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        wandb_step = int(metrics["epoch"])
        metrics = self._add_prefix(metrics)
        if step is not None:
            self.experiment.log({**metrics, "trainer/global_step": step},
                                step=wandb_step)
        else:
            self.experiment.log(metrics, step=wandb_step)

    @property
    def name(self) -> Optional[str]:
        """ Override the method because model checkpointing define the path before
        the initialization, and in offline mode you can't get the good path
        """
        # don't create an experiment if we don't have one
        # return self._experiment.project_name() if self._experiment else self._name
        return self._wandb_init["project"]

    def symlink_checkpoint(self, code_dir, project, run_id):
        local_project_dir = Path("wandb") / project
        local_project_dir.mkdir(parents=True, exist_ok=True)
        Path(code_dir) / project / run_id
        os.symlink(Path(code_dir) / "wandb" / project / run_id, local_project_dir / run_id)
        # Creating a another symlink for easy access
        os.symlink(Path(code_dir) / "wandb" / project / run_id / "checkpoints", Path("checkpoints"))

    def symlink_run(self, checkpoint_folder: str):
        code_dir = checkpoint_folder.split("wandb/")[0]
        # local run
        local_wandb = Path("wandb/wandb")
        local_wandb.mkdir(parents=True, exist_ok=True)
        offline_run = self.experiment.dir.split("wandb/wandb/")[1].split("/files")[0]

        # Create the symlink
        os.symlink(Path(code_dir) / "wandb/wandb" / offline_run, local_wandb / offline_run)

    def begin(self, code_dir, project, run_id):
        self.symlink_checkpoint(code_dir, project, run_id)

    def end(self, checkpoint_folder):
        self.symlink_run(checkpoint_folder)
