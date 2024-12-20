import numpy as np
import imageio
from utils import greedy_policy

max_count = 1000

def record_video(env, Qtable, out_directory, fps=1):
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=np.random.randint(0, 500))
    img = env.render()

    # TODO: This count is to end if there is an infinite loop, fix code to not need this later
    count = 0

    while (not terminated or truncated) and count < max_count:
        action = greedy_policy(Qtable, state)
        state, _, terminated, truncated, _ = env.step(action)
        img = env.render()
        images.append(img)
        count += 1
        
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
