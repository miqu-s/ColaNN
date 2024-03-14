import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from model import ImgModel

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.color_paths = sorted(os.listdir(os.path.join(root_dir, 'color')))
        self.depth_paths = sorted(os.listdir(os.path.join(root_dir, 'depth')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        color_img_path = os.path.join(self.root_dir, 'color', self.color_paths[idx])
        depth_img_path = os.path.join(self.root_dir, 'depth', self.depth_paths[idx])

        color_img = Image.open(color_img_path).convert("RGB")
        depth_img = Image.open(depth_img_path).convert("RGB")

        color_img = self.transform(color_img)
        depth_img = self.transform(depth_img)

        return color_img, depth_img

color_channels = 3
depth_channels = 3
cnn_channels = [64, 128, 256, 512, 1024, 2048, 4096]

img_model = ImgModel(img_size=248, cnn_channels=cnn_channels, color_channels=color_channels, depth_channels=depth_channels)

criterion = nn.MSELoss()
optimizer = optim.Adam(img_model.parameters(), lr=1e-2)

dataset_path = 'dataset/'

dataset = CustomDataset(root_dir=dataset_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 50
inference_interval = 1
device = torch.device("dml" if torch.cuda.is_available() else "cpu")
img_model.to(device)

for epoch in range(num_epochs):
    for batch_color, batch_depth in dataloader:
        batch_color, batch_depth = batch_color.to(device), batch_depth.to(device)

        output = img_model(batch_color, batch_depth)

        loss = criterion(output, batch_color)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    if (epoch + 1) % inference_interval == 0:
        input_color = Image.open(os.path.join(dataset_path, 'color/cube_color.png')).convert("RGB")
        input_depth = Image.open(os.path.join(dataset_path, 'depth/cube_depth.png')).convert("RGB")

        input_color = transforms.ToTensor()(input_color).unsqueeze(0).to(device)
        input_depth = transforms.ToTensor()(input_depth).unsqueeze(0).to(device)

        with torch.no_grad():
            img_model.eval()
            inferred_output = img_model(input_color, input_depth)
            img_model.train()

        inferred_output = inferred_output.squeeze().cpu().numpy()
        inferred_output = (inferred_output * 255).astype('uint8')
        inferred_output_img = Image.fromarray(inferred_output.transpose(1, 2, 0))

        inferred_output_img.save(f'rendered/inference_epoch_{epoch+1}.png')

torch.save(img_model.state_dict(), 'model.pth')
