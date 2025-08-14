# A Attention Visualization Tool
from matplotlib import pyplot as plt
import os
def plot_attention(attention_weights, multi_head, id, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if multi_head > 1:
        plt.figure(figsize=(10*multi_head//2, 10*2))
        for head in range(multi_head):
            plt.subplot(2, multi_head//2, head + 1)
            plt.imshow(attention_weights[head], aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Head {head}')
            plt.xlabel('Decoder Time Steps')
            plt.ylabel('Encoder Time Steps')
        plt.suptitle(f'Attention Weights - {id}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{save_path}/attention_weights_{id}_multihead.png')
        plt.close()
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(attention_weights[0], aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights - {id}')
        plt.xlabel('Decoder Time Steps')
        plt.ylabel('Encoder Time Steps')
        plt.savefig(f'{save_path}/attention_weights_{id}.png')
        plt.close()