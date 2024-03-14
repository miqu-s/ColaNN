import torch

from torchvision import transforms

from PIL import Image

from model import ImgModel


img_model = ImgModel(img_size=248)

img_model.load_state_dict(torch.load('model.pth'))

img_model.eval()


input_color_path = r'C:\Users\mikus\Desktop\render\dataset\color\sphere_color.png'

input_depth_path = r'C:\Users\mikus\Desktop\render\dataset\depth\sphere_depth.png'


input_color = Image.open(input_color_path).convert("RGB")

input_depth = Image.open(input_depth_path).convert("RGB")


transform = transforms.Compose([

    transforms.ToTensor(),

])



input_color = transform(input_color).unsqueeze(0)

input_depth = transform(input_depth).unsqueeze(0)


with torch.no_grad():

    output = img_model(input_color, input_depth)



output = output.squeeze().cpu().numpy()

output = (output * 255).astype('uint8')

output_img = Image.fromarray(output.transpose(1, 2, 0))



output_img.save('inferred_output.png')

