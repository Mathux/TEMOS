import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BASEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int):
        super().__init__()

        from temos.data.tools import collate_datastruct_and_text
        self.dataloader_options = {"batch_size": batch_size, "num_workers": num_workers,
                                   "collate_fn": collate_datastruct_and_text}
        # need to be overloaded:
        # - self.Dataset
        # - self._sample_set => load only a small subset
        #   There is an helper bellow (get_sample_set)
        # - self.nfeats
        # - self.transforms

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.Dataset(**sample_params)

    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                self.__dict__[item_c] = self.Dataset(split=subset, **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_options)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_options)
