from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os


def create_frame(image, frame_number, probs, action_taken):
    border_size = 60
    new_size = (image.width, image.height + border_size)
    new_frame = Image.new("RGB", new_size, "white")
    new_frame.paste(image, ((new_size[0] - image.width) // 2, 0))

    draw = ImageDraw.Draw(new_frame)
    font = ImageFont.truetype("DejaVuSans.ttf", size = 7)

    # draw.text((10, 10), f"Step {frame_number}", fill = "black", font = font)
    # draw.text((10, new_size[1] - 60), f"0: {probs[0]:.2f}", fill = "black", font = font)
    # draw.text((10, new_size[1] - 45), f"1: {probs[1]:.2f}", fill = "black", font = font)
    
    if action_taken == 1:
        text = "Help!"
        text_width, text_height = draw.textsize(text, font = font)
        draw.text(((new_size[0] - text_width) // 2, image.height + 5), text, fill = "red", font = font)
    
    return new_frame


def create_gif(env_data, env_index, save_dir):
    frames = []
    for step in range(len(env_data["observations"])):
        obs = env_data["observations"][step].permute(1, 2, 0)
        obs = (obs * 255).cpu().numpy().astype(np.uint8)
        image = Image.fromarray(obs)

        probs = env_data["probs"][step]
        action_taken = env_data["actions"][step]
        
        image = create_frame(image, step, probs, action_taken)
        frames.append(image)
    gif_path = os.path.join(save_dir, f"env_{env_index}_trajectory.gif")
    frames[0].save(gif_path, save_all = True, append_images = frames[1:], duration = 250, loop = 0)
    print(f"GIF {env_index} saved at {gif_path}")


def animate(env_data, save_dir):
    for env_index in env_data:
        create_gif(env_data[env_index], env_index, save_dir)
