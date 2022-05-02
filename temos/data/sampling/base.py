from typing import Optional
from dataclasses import dataclass


@dataclass
class FrameSampler:
    sampling: str = "conseq"
    sampling_step: int = 1
    request_frames: Optional[int] = None
    threshold_reject: int = 0.75
    max_len: int = 1000
    min_len: int = 10

    def __call__(self, num_frames):
        from .frames import get_frameix_from_data_index
        return get_frameix_from_data_index(num_frames,
                                           self.request_frames,
                                           self.sampling,
                                           self.sampling_step)

    def accept(self, duration):
        # Outputs have original lengths
        # Check if it is too long
        if self.request_frames is None:
            if duration > self.max_len:
                return False
            elif duration < self.min_len:
                return False
        else:
            # Reject sample if the length is
            # too little relative to
            # the request frames
            min_number = self.threshold_reject * self.request_frames
            if duration < min_number:
                return False
        return True

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)
