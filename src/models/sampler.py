from src.models.score_models import ScoreModule
import torch
from diffusers import DDPMScheduler
from src.utils.dataclasses import DiffusableBatch
from typing import Optional, Union, List



class TSSampler():
    def __init__(self, score_model: ScoreModule, scheduler: DDPMScheduler, device: str = "cpu") -> None:
        
        self.score_model = score_model.to(device)
        self.scheduler = scheduler
    
    def __call__(self, batch_size: int = 1, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, num_inference_steps: int = 1000) -> None:
        
        #Extract dimensions from the score model
        ts_shape = (batch_size, self.score_model.max_len, self.score_model.n_channels)
        
        #Generate noise samples (t = T )
        X = torch.randn(ts_shape)
        
        
        #Set the time steps for the scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            print(t)
            # Create a DiffusableBatch
            timesteps=(torch.ones(batch_size, dtype=torch.long) * t).to(X.device)
            batch = DiffusableBatch(X, timesteps=timesteps)
            # 1. predict noise model_output
            denoised_ts = self.score_model(batch)

            # 2. compute previous image: x_t -> x_t-1
            X = self.scheduler.step(denoised_ts, t, X, generator=generator).prev_sample

        
        return X