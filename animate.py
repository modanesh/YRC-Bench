from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import os
import torch
import argparse

# ref: logs/procgen/coinrun_aisc/help_test_ent_og_risk_10/RENDER_seed_8888_06-25-2024_17-42-18/obs_env_0_seed_8888_step_40.png, pixels 30-40 for x, y
# R: 1, 0.8902
# G: 0.8, 0.7137
# B: 0, 0.0078

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
    coin_presents = []
    ask_for_helps = []
    try:
        for step in range(len(env_data["observations"])):
            obs = torch.from_numpy(env_data["observations"][step]).permute(1, 2, 0)
            obs = (obs * 255).cpu().numpy().astype(np.uint8)
            # image = Image.fromarray(obs)

            RED, GREEN, BLUE = 0, 1, 2
            reds = [int(val * 255) for val in [1, 0.8902]]
            greens = [int(val * 255) for val in [0.8, 0.7137]]
            blues = [int(val * 255) for val in [0, 0.0078]]
            height, width, _ = obs.shape
            coin_present = False
            for y in range(height):
                for x in range(width):
                    if obs[y, x, RED] in reds:
                        if obs[y, x, GREEN] in greens:
                            if obs[y, x, BLUE] in blues:
                                coin_present = True
            coin_presents.append(coin_present)

            probs = env_data["probs"][step]
            action_taken = env_data["actions"][step]
            ask_for_helps.append(action_taken)
        
            # image = create_frame(image, step, probs, action_taken)
            # frames.append(image)
    except IndexError:
        pass
    # gif_path = os.path.join(save_dir, f"env_{env_index}_trajectory.gif")
    # frames[0].save(gif_path, save_all = True, append_images = frames[1:], duration = 250, loop = 0)
    # print(f"GIF {env_index} saved at {gif_path}")
    min_length = min(len(coin_presents), len(ask_for_helps))
    coin_presents = np.array(coin_presents).astype(int)[:min_length]
    ask_for_helps = np.array(ask_for_helps).astype(int)[:min_length]
    no_coin_and_help = (coin_presents == 0) & (ask_for_helps == 1)
    no_coin = (coin_presents == 0)
    p_ask_help_when_no_coin = no_coin_and_help.sum() / no_coin.sum()
    yes_coin_and_help = (coin_presents == 1) & (ask_for_helps == 1)
    yes_coin = (coin_presents == 1)
    p_ask_help_when_yes_coin = yes_coin_and_help.sum() / yes_coin.sum()
    return p_ask_help_when_no_coin, p_ask_help_when_yes_coin



def animate(env_data, save_dir):
    p_no_coin_avg, p_yes_coin_avg = [], []
    for env_index in env_data:
        p_no_coin, p_yes_coin = create_gif(env_data[env_index], env_index, save_dir)
        p_no_coin_avg.append(p_no_coin)
        p_yes_coin_avg.append(p_yes_coin)
    print("NO COIN ASK FOR HELP:", np.mean(p_no_coin_avg))
    print("YES COIN ASK FOR HELP:", np.mean(p_yes_coin_avg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type = str)
    args = parser.parse_args()
    full_dir = os.path.join("logs/coinrun_aisc/", args.dir)
    with open(os.path.join(full_dir, "gif_data.pkl"), "rb") as f:
        env_data = pickle.load(f)
    animate(env_data, full_dir)
