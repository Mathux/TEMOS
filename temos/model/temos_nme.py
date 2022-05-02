from typing import List, Optional

import torch
from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from temos.model.utils.tools import remove_padding

from temos.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from temos.model.base import BaseModel


class TEMOS_nme(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 nvids_to_save: Optional[int] = None,
                 **kwargs):
        super().__init__()

        self.textencoder = instantiate(textencoder)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)

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
        datastruct_from_text = self.text_to_motion_forward(batch["text"],
                                                           batch["length"])

        return remove_padding(datastruct_from_text.joints, batch["length"])

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int],
                               return_latent: bool = False
                               ):
        # Encode the text to the latent space
        if self.hparams.vae:
            distribution = self.textencoder(text_sentences)

            if self.sample_mean:
                latent_vector = distribution.loc
            else:
                # Reparameterization trick
                eps = distribution.rsample() - distribution.loc
                latent_vector = distribution.loc + self.fact * eps
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        datastruct_from_text, latent_from_text, distribution_from_text = ret

        # GT data
        datastruct_ref = batch["datastruct"]

        # Compare to a Normal distribution
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None

        # Compute the losses
        loss = self.losses[split].update(ds_text=datastruct_from_text,
                                         ds_motion=None,
                                         ds_ref=datastruct_ref,
                                         lat_text=latent_from_text,
                                         lat_motion=None,
                                         dis_text=distribution_from_text,
                                         dis_motion=None,
                                         dis_ref=distribution_ref)
        if split == "val":
            # Compute the metrics
            self.metrics.update(datastruct_from_text.detach().joints,
                                datastruct_ref.detach().joints,
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
                    "ref": prepare(datastruct_ref),
                    "from_text": prepare(datastruct_from_text)
                }

        return loss
