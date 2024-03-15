import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgModel(nn.Module):
    def __init__(self, img_size, color_channels, depth_channels, cnn_channels):
        super(ImgModel, self).__init__()
        color_cnn_layers = [
            nn.Conv2d(color_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ]

        self.color_cnn = nn.Sequential(*color_cnn_layers)

        depth_cnn_layers = [
            nn.Conv2d(depth_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ]

        self.depth_cnn = nn.Sequential(*depth_cnn_layers)

        self.merged_cnn = nn.Sequential(
            nn.Conv2d(2 * cnn_channels[1], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(cnn_channels[1], cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.output_layer = nn.Conv2d(cnn_channels[0], 3, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, color_input, depth_input):
        c_cnn = self.color_cnn(color_input)
        d_cnn = self.depth_cnn(depth_input)

        c_cnn = self.dropout(c_cnn)
        d_cnn = self.dropout(d_cnn)
        
        merged = torch.cat([c_cnn, d_cnn], dim=1)
        merged = self.merged_cnn(merged)
        output = self.output_layer(merged)

        return output
