from typing import List, Optional

import torch
import numpy as np
from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from temos.data.tools import PoseData
from temos.model.utils.tools import remove_padding

from temos.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from temos.model.base import BaseModel


class ActionTEMOS(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 pose2joints: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 nvids_to_save: Optional[int] = None,
                 **kwargs):
        super().__init__()

        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder, nfeats=nfeats)

        pose2joints = instantiate(pose2joints)
        self.pose2joints = pose2joints
        self.motiondecoder = instantiate(motiondecoder,
                                         nfeats=nfeats,
                                         pose2joints=pose2joints)

        self._losses = MetricCollection({split: instantiate(losses, vae=vae,
                                                            _recursive_=False)
                                         for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}

        self.metrics = ComputeMetrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict) -> List[Tensor]:
        # convert text into actions
        pose_data_from_text = self.actions_to_motion_forward(batch["actions"],
                                                             batch["length"])

        return remove_padding(pose_data_from_text.joints, batch["length"])

    def actions_to_motion_forward(self, text_sentences: List[str], lengths: List[int],
                                  return_latent: bool = False):
        # Encode the text to the latent space
        if self.hparams.vae:
            distributions = self.textencoder(text_sentences)

            latent_vectors = []
            if self.sample_mean:
                for distribution in distributions:
                    latent_vector = distribution.loc
                    latent_vectors.append(latent_vector)
            else:
                for distribution in distributions:
                    # Reparameterization trick
                    eps = distribution.rsample() - distribution.loc
                    latent_vector = distribution.loc + self.fact * eps
                    latent_vectors.append(latent_vector)
        else:
            distributions = None
            latent_vectors = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        pose_data: PoseData = self.motiondecoder(latent_vectors, lengths)

        if not return_latent:
            return pose_data
        return pose_data, latent_vectors, distributions

    def motion_to_motion_forward(self, pose_data: PoseData,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False):
        # Make sure it is on the good device
        pose_data.pose2joints = self.pose2joints

        # Encode the motion to the latent space
        if self.hparams.vae:
            distribution = self.motionencoder(pose_data.features, lengths)

            if self.sample_mean:
                latent_vector = distribution.loc
            else:
                # Reparameterization trick
                eps = distribution.rsample() - distribution.loc
                latent_vector = distribution.loc + self.fact * eps
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(pose_data.features, lengths)

        # Decode the latent vector to a motion
        pose_data: PoseData = self.motiondecoder(latent_vector, lengths)

        if not return_latent:
            return pose_data
        return pose_data, latent_vector, distribution

    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        ret = self.actions_to_motion_forward(batch["actions"],
                                             batch["length"],
                                             return_latent=True)
        pose_data_from_text, latents_from_text, distributions_from_text = ret

        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(batch["pose_data"],
                                            batch["length"],
                                            return_latent=True)
        pose_data_from_motion, latent_from_motion, distribution_from_motion = ret

        # GT data
        pose_data_ref = batch["pose_data"]

        # Compare each distribution to a normal distribution
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            distributions_ref = [
                torch.distributions.Normal(torch.zeros_like(dist.loc),
                                           torch.ones_like(dist.scale))
                for dist in distributions_from_text]
            distribution_ref = torch.distributions.Normal(
                torch.zeros_like(distribution_from_motion.loc),
                torch.ones_like(distribution_from_motion.scale))
        else:
            distributions_ref = None
            distribution_ref = None

        # Compute the losses
        loss = self.losses[split].update(pd_text=pose_data_from_text,
                                         pd_motion=pose_data_from_motion,
                                         pd_ref=pose_data_ref,
                                         lats_text=latents_from_text,
                                         lat_motion=latent_from_motion,
                                         diss_text=distributions_from_text,
                                         dis_motion=distribution_from_motion,
                                         dis_ref=distribution_ref,
                                         diss_ref=distributions_ref)
        if split == "val":
            # Compute the metrics
            self.metrics.update(pose_data_from_text.detach().joints,
                                pose_data_ref.detach().joints,
                                batch["length"])

        if batch_idx == 0:
            nvids = self.hparams.nvids_to_save
            if nvids is not None and nvids != 0:
                del self.store_examples[split]
                lengths = batch["length"][:nvids]

                def prepare(x):
                    x = x.detach().joints[:nvids]
                    x = x.cpu().numpy()
                    return remove_padding(x, lengths)

                self.store_examples[split] = {
                    "text": batch["text"][:nvids],
                    "ref": prepare(pose_data_ref),
                    "from_text": prepare(pose_data_from_text),
                    "from_motion": prepare(pose_data_from_motion)
                }

        return loss


"""
class ActionLevel(LightningModule):
    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        pose_data_from_text = self.actions_to_motion_forward(batch["actions"],
                                                             batch["length"])
        # GT data
        pose_data_ref = batch["pose_data"]

        # Compute the losses
        loss = self.losses[split].update(pd_text=pose_data_from_text,
                                         pd_ref=pose_data_ref)
        if split == "val":
            # Compute the metrics
            self.metrics.update(pose_data_from_text.detach().joints,
                                pose_data_ref.detach().joints,
                                batch["length"])

        return loss

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def allsplit_epoch_end(self, split: str, outputs):
        losses = self.losses[split]
        loss_dict = losses.compute(split)
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}

        if split == "val":
            metrics_dict = self.metrics.compute()
            dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})
        dico.update({"epoch": float(self.trainer.current_epoch),
                     "step": float(self.trainer.current_epoch)})
        self.log_dict(dico)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self.allsplit_epoch_end("test", outputs)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optim, params=self.parameters())
        return {"optimizer": optimizer}
"""
