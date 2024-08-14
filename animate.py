from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--dir", "-d", type = str)
args = parser.parse_args()


def create_frame(image, frame_number, probs, action_taken):
    border_size = 60
    new_size = (image.width + 2 * border_size, image.height + border_size + border_size // 2)

    new_frame = Image.new("RGB", new_size, "white")
    new_frame.paste(image, (border_size, border_size // 3))

    draw = ImageDraw.Draw(new_frame)
    font = ImageFont.truetype("DejaVuSans.ttf", size = 7)

    if action_taken == 1:
        one_action = probs[0]
        zero_action = probs[1]
    elif action_taken == 0:
        one_action = probs[1]
        zero_action = probs[0]

    draw.text((10, 10), f"Step {frame_number}", fill = "black", font = font)
    draw.text((10, new_size[1] - 60), f"0: {zero_action:.2f}", fill = "black", font = font)
    draw.text((10, new_size[1] - 45), f"1: {one_action:.2f}", fill = "black", font = font)
    
    if action_taken == 1:
        draw.text((10, new_size[1] - 30), "Asked for help!", fill = "green", font = font)
    
    return new_frame


def create_gif(env_data, env_index, save_dir):
    frames = []
    try:
        for step in range(len(env_data["actions"])):
            obs = torch.from_numpy(env_data["observations"][step]).permute(1, 2, 0)
            obs = (obs * 255).cpu().numpy().astype(np.uint8)
            image = Image.fromarray(obs)

            probs = env_data["probs"][step]
            action_taken = env_data["actions"][step]
        
            image = create_frame(image, step, probs, action_taken)
            frames.append(image)
    except IndexError:
        pass
    gif_path = os.path.join(save_dir, f"env_{env_index}_trajectory.gif")
    frames[0].save(gif_path, save_all = True, append_images = frames[1:], duration = 250, loop = 0)
    print(f"GIF {env_index} saved at {gif_path}")


def animate(env_data, save_dir):
    for env_index in env_data:
        create_gif(env_data[env_index], env_index, save_dir)

with open(os.path.join("/nas/ucb/tutrinh/yield_request_control/logs/coinrun_aisc/", args.dir, "gif_data.pkl"), "rb") as f:
    env_data = pickle.load(f)
animate(env_data, os.path.join("/nas/ucb/tutrinh/yield_request_control/logs/coinrun_aisc/", args.dir))

