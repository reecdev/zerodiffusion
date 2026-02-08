import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

path = input("Enter image path: ")
img = Image.open(path).convert("RGB").resize((512, 512))
t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().cuda() / 127.5 - 1.0

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").cuda()
with torch.no_grad():
    lat = vae.encode(t.unsqueeze(0)).latent_dist.sample() * 0.18215

del vae
torch.cuda.empty_cache()

model = Detector().cuda()
model.load_state_dict(torch.load("detector.pth"))
model.eval()

with torch.no_grad():
    logits = model(lat)
    probs = F.softmax(logits, dim=1)

probreal = probs[0][0].item()*100
probfake = probs[0][1].item()*100

print(f"Real: {probreal:.2f}%")
print(f"Fake: {probfake:.2f}%")

print(f"Label: {'Real' if probreal > probfake else 'Fake'}")