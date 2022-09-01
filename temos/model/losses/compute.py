import hydra
import torch

from torch.nn import Module

class TemosComputeLosses(Module):
    def __init__(self, vae: bool,
                 mode: str,
                 loss_on_both: bool = False,
                 force_loss_on_jfeats: bool = True,
                 ablation_no_kl_combine: bool = False,
                 ablation_no_motionencoder: bool = False,
                 ablation_no_kl_gaussian: bool = False, **kwargs):
        super().__init__()

        # Save parameters
        self.vae = vae
        self.mode = mode

        self.loss_on_both = loss_on_both
        self.force_loss_on_jfeats = force_loss_on_jfeats
        self.ablation_no_kl_combine = ablation_no_kl_combine
        self.ablation_no_kl_gaussian = ablation_no_kl_gaussian
        self.ablation_no_motionencoder = ablation_no_motionencoder

        losses = []
        if mode == "xyz" or force_loss_on_jfeats:
            if not ablation_no_motionencoder:
                losses.append("recons_jfeats2jfeats")
            losses.append("recons_text2jfeats")
        if mode == "smpl":
            if not ablation_no_motionencoder:
                losses.append("recons_rfeats2rfeats")
            losses.append("recons_text2rfeats")
        else:
            ValueError("This mode is not recognized.")

        if vae or loss_on_both:
            kl_losses = []
            if not ablation_no_kl_combine and not ablation_no_motionencoder:
                kl_losses.extend(["kl_text2motion", "kl_motion2text"])
            if not ablation_no_kl_gaussian:
                if ablation_no_motionencoder:
                    kl_losses.extend(["kl_text"])
                else:
                    kl_losses.extend(["kl_text", "kl_motion"])
            losses.extend(kl_losses)
        if not self.vae or loss_on_both:
            if not ablation_no_motionencoder:
                losses.append("latent_manifold")
        losses.append("total")

        self.losses_values = {}
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))

        self.register_buffer("count", torch.tensor(0.0))
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: hydra.utils.instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss != "total"}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, ds_text=None, ds_motion=None, ds_ref=None,
               lat_text=None, lat_motion=None, dis_text=None,
               dis_motion=None, dis_ref=None):
        total: float = 0.0

        if self.mode == "xyz" or self.force_loss_on_jfeats:
            if not self.ablation_no_motionencoder:
                total += self._update_loss("recons_jfeats2jfeats", ds_motion.jfeats, ds_ref.jfeats)
            total += self._update_loss("recons_text2jfeats", ds_text.jfeats, ds_ref.jfeats)

        if self.mode == "smpl":
            if not self.ablation_no_motionencoder:
                total += self._update_loss("recons_rfeats2rfeats", ds_motion.rfeats, ds_ref.rfeats)
            total += self._update_loss("recons_text2rfeats", ds_text.rfeats, ds_ref.rfeats)

        if self.vae or self.loss_on_both:
            if not self.ablation_no_kl_combine and not self.ablation_no_motionencoder:
                total += self._update_loss("kl_text2motion", dis_text, dis_motion)
                total += self._update_loss("kl_motion2text", dis_motion, dis_text)
            if not self.ablation_no_kl_gaussian:
                total += self._update_loss("kl_text", dis_text, dis_ref)
                if not self.ablation_no_motionencoder:
                    total += self._update_loss("kl_motion", dis_motion, dis_ref)
        if not self.vae or self.loss_on_both:
            if not self.ablation_no_motionencoder:
                total += self._update_loss("latent_manifold", lat_text, lat_motion)

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = self.count
        # return {loss: self.losses_values[loss]/count for loss in self.losses}
        return {loss: getattr(self, loss)/count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        # self.losses_values[loss] += val.detach()
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
