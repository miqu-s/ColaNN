from PIL import Image, ImageDraw, ImageFont
import os

def create_gif(image_folder, output_gif_path, fps=60, font_size=20):
    images = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))])

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), image_file, fill="white", font=font)

        images.append(img)

    images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=int(100/fps), loop=0)

if __name__ == "__main__":
    image_folder_path = "rendered"
    output_gif_path = "animated_output.gif"

    create_gif(image_folder_path, output_gif_path)
    print(output_gif_path)
