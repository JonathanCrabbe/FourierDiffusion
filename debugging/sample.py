# append to the path
import sys

sys.path.append("/home/nvth2/FourierDiffusion/src")
sys.path.append("/home/nvth2/FourierDiffusion")
import matplotlib.pyplot as plt
import torch
from src.models.sampler import TSSampler
from src.models.score_models import ScoreModule


def visualize_ts(ts: torch.Tensor, title: str = "Time series") -> None:
    fig, ax = plt.subplots()
    ax.plot(ts.detach().cpu().numpy())
    ax.set_title(title)
    # save the plot
    fig.savefig("/home/nvth2/FourierDiffusion/debugging/TimeSeries.png")
    plt.show()


# Load the checkpoint

score_model = ScoreModule.load_from_checkpoint(
    "/home/nvth2/FourierDiffusion/cmd/lightning_logs/x58vjcza/checkpoints/epoch=23-val_loss=0.02.ckpt"
)
score_model.eval()
sampler = TSSampler(score_model, score_model.noise_scheduler)
ts = sampler(batch_size=2, num_inference_steps=1000)

visualize_ts(ts[0], title="Time series")
