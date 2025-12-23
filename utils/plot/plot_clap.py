## CLAP-like audio/phoneme clustering plot functions

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

def plot_phoneme_centroids_with_instances(audio_feats, phoneme_feats, phoneme_labels, 
                                         phone_scores=None, ignore_index=None, max_phones=5,
                                         show_audio_scatter=True, show_audio_centroid=True,
                                         show_phoneme_scatter=True, show_phoneme_centroid=True):
    """
    Plots phoneme features as centroids (centers marked with 'x') with all instances around them.
    Audio and phoneme features are both compressed to 2D using t-SNE.
    Audio features are shown in Tokyo University deep blue, phoneme features are shown in gold.
    Returns two plots: main plot with all phonemes, and zoom-in plot with 4 phonemes using different markers.
    
    Parameters:
    - audio_feats: [B, T, D]
    - phoneme_feats: [B, T, D]
    - phoneme_labels: [B, T]
    - phone_scores: [B, T]  score range [0, 2] (optional)
    - ignore_index: int, list, or tuple of indices to ignore in phoneme_labels (default: None)
    - max_phones: int, maximum number of phonemes to plot
    - show_audio_scatter: bool, whether to show audio feature instances (default: True)
    - show_audio_centroid: bool, whether to show audio centroid (default: True)
    - show_phoneme_scatter: bool, whether to show phoneme feature instances (default: True)
    - show_phoneme_centroid: bool, whether to show phoneme centroid (default: True)
    return fig_main, ax_main, fig_zoom, ax_zoom
    """
    from sklearn.manifold import TSNE
    
    # Flatten the features and labels
    B, T, D = audio_feats.shape
    audio_feats_flat = audio_feats.reshape(-1, D)
    phoneme_feats_flat = phoneme_feats.reshape(-1, D)
    phoneme_labels_flat = phoneme_labels.reshape(-1)
    
    if phone_scores is not None:
        phone_scores_flat = phone_scores.reshape(-1)
    
    # Convert ignore_index to a set for efficient lookup
    if ignore_index is None:
        ignore_indices = set()
    elif isinstance(ignore_index, (list, tuple)):
        ignore_indices = set(ignore_index)
    else:
        ignore_indices = {ignore_index}
    
    # Filter out ignore_indices
    valid_indices = ~np.isin(phoneme_labels_flat, list(ignore_indices))
    audio_feats_valid = audio_feats_flat[valid_indices]
    phoneme_feats_valid = phoneme_feats_flat[valid_indices]
    phoneme_labels_valid = phoneme_labels_flat[valid_indices]
    
    if phone_scores is not None:
        phone_scores_valid = phone_scores_flat[valid_indices]
    
    # Get the most frequent phonemes
    unique, counts = np.unique(phoneme_labels_valid, return_counts=True)
    freq_phonemes = unique[np.argsort(-counts)][:max_phones]
    top_4_phonemes = unique[np.argsort(-counts)][:4]
    
    # Combine and compress features using t-SNE on all valid data
    all_feats = np.vstack([audio_feats_valid, phoneme_feats_valid])
    
    print(f"Compressing {all_feats.shape[0]} features to 2D using t-SNE...")
    # import pdb; pdb.set_trace()
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(all_feats) // 2))
    all_feats_2d = tsne.fit_transform(all_feats)
    
    # Split back into audio and phoneme features
    audio_feats_2d = all_feats_2d[:len(audio_feats_valid)]
    phoneme_feats_2d = all_feats_2d[len(audio_feats_valid):]
    
    # ===== MAIN PLOT: All phonemes with blue audio and orange phoneme =====
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot instances and calculate centroids for each phoneme
    for phn in freq_phonemes:
        phn_indices = phoneme_labels_valid == phn
        
        # Plot audio feature instances (dots) in blue
        if show_audio_scatter:
            ax.scatter(
                audio_feats_2d[phn_indices, 0],
                audio_feats_2d[phn_indices, 1],
                c='blue',
                marker='o',
                s=50,
                alpha=1.0,
                edgecolors='none'
            )
        
        # Plot phoneme feature instances (dots) in orange
        if show_phoneme_scatter:
            ax.scatter(
                phoneme_feats_2d[phn_indices, 0],
                phoneme_feats_2d[phn_indices, 1],
                c='orange',
                marker='s',
                s=50,
                alpha=1.0,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Calculate and plot centroid for audio features
        audio_centroid = audio_feats_2d[phn_indices].mean(axis=0)
        if show_audio_centroid:
            ax.scatter(
                audio_centroid[0],
                audio_centroid[1],
                c='blue',
                marker='x',
                s=200,
                linewidths=3,
                edgecolors='black',
                label=f'Phoneme {phn} (Audio)' if phn == freq_phonemes[0] else ''
            )
        
        # Calculate and plot centroid for phoneme features
        phoneme_centroid = phoneme_feats_2d[phn_indices].mean(axis=0)
        if show_phoneme_centroid:
            ax.scatter(
                phoneme_centroid[0],
                phoneme_centroid[1],
                c='orange',
                marker='+',
                s=300,
                linewidths=3,
                edgecolors='black',
                label=f'Phoneme {phn} (Phoneme)' if phn == freq_phonemes[0] else ''
            )
        
        # Draw line between centroids (only if both centroids are shown)
        if show_audio_centroid and show_phoneme_centroid:
            ax.plot(
                [audio_centroid[0], phoneme_centroid[0]],
                [audio_centroid[1], phoneme_centroid[1]],
                color='gray',
                linestyle='--',
                alpha=1.0,
                linewidth=1
            )
    
    ax.set_title('Phoneme Centroids with Feature Instances (All Phonemes)', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(False)
    
    # Add legend for main plot
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Audio Features', alpha=0.8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, label='Phoneme Features', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='x', color='blue', linewidth=0, markersize=10, markeredgewidth=2, label='Audio Centroid'),
        Line2D([0], [0], marker='+', color='orange', linewidth=0, markersize=10, markeredgewidth=2, label='Phoneme Centroid'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    plt.tight_layout()
    
    # ===== ZOOM-IN PLOT: Top 4 phonemes with different markers for each =====
    markers = ['o', 's', '^', 'x']  # circle, square, triangle, x
    marker_names = ['Circle', 'Square', 'Triangle', 'X']
    colors_zoom = ['red', 'green', 'orange', 'purple']
    
    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 10))
    
    for idx, phn in enumerate(top_4_phonemes):
        phn_indices = phoneme_labels_valid == phn
        marker = markers[idx]
        color = colors_zoom[idx]
        marker_name = marker_names[idx]
        
        # Plot audio feature instances with current marker in blue
        if show_audio_scatter:
            ax_zoom.scatter(
                audio_feats_2d[phn_indices, 0],
                audio_feats_2d[phn_indices, 1],
                c='blue',
                marker=marker,
                s=80,
                alpha=0.4
            )
        
        # Plot phoneme feature instances with current marker in orange
        if show_phoneme_scatter:
            ax_zoom.scatter(
                phoneme_feats_2d[phn_indices, 0],
                phoneme_feats_2d[phn_indices, 1],
                c='orange',
                marker=marker,
                s=80,
                alpha=0.6
            )
        
        # Calculate and plot centroid for audio features
        audio_centroid = audio_feats_2d[phn_indices].mean(axis=0)
        if show_audio_centroid:
            ax_zoom.scatter(
                audio_centroid[0],
                audio_centroid[1],
                c='blue',
                marker=marker,
                s=300,
                zorder=10
            )
        
        # Calculate and plot centroid for phoneme features
        phoneme_centroid = phoneme_feats_2d[phn_indices].mean(axis=0)
        if show_phoneme_centroid:
            ax_zoom.scatter(
                phoneme_centroid[0],
                phoneme_centroid[1],
                c='orange',
                marker=marker,
                s=300,
                zorder=10
            )
        
        # Draw line between centroids (only if both centroids are shown)
        if show_audio_centroid and show_phoneme_centroid:
            ax_zoom.plot(
                [audio_centroid[0], phoneme_centroid[0]],
                [audio_centroid[1], phoneme_centroid[1]],
                color=color,
                linestyle='--',
                alpha=1.0,
                linewidth=2
            )
    
    ax_zoom.set_title('Phoneme Centroids Zoom-In (Top 4 Phonemes)', fontsize=16, fontweight='bold')
    ax_zoom.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax_zoom.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax_zoom.grid(True, alpha=0.3)
    
    # Add legend for zoom plot
    zoom_legend_elements = []
    for idx, phn in enumerate(top_4_phonemes):
        marker = markers[idx]
        marker_name = marker_names[idx]
        zoom_legend_elements.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='blue', markersize=10, 
                   label=f'Phoneme {phn} ({marker_name})', markeredgecolor='darkblue', markeredgewidth=1)
        )
    zoom_legend_elements.append(Line2D([0], [0], marker='', color='blue', linewidth=2, label='Audio (Blue)'))
    zoom_legend_elements.append(Line2D([0], [0], marker='', color='orange', linewidth=2, label='Phoneme (orange)', markeredgecolor='black'))
    ax_zoom.legend(handles=zoom_legend_elements, loc='best', fontsize=10)
    plt.tight_layout()
    
    return fig, ax, fig_zoom, ax_zoom


def plot_clap_clusters(audio_feats, phoneme_feats, phoneme_labels, phone_scores, ignore_index=None, max_phones=5):
    """
    Plots the clustering of audio and phoneme features in a tsn-e 2D space.
    If it follows CLAP's approach, it should visualizes how audio features align with phoneme features.
    Plot the most frequent phonemes and it's realated audio_feats up to max_phones
    Use x for phoneme features and . for audio features, with different colors for different phonemes.

    Parameters:
    - audio_feats: [B, T, D]
    - phoneme_feats: [B, T, D]
    - phoneme_labels: [B, T]
    - phone_scores: [B, T]  score range [0, 2]
    - ignore_index: int, list, or tuple of indices to ignore in phoneme_labels (default: None)
    - max_phones: int, maximum number of phonemes to plot
    return fig, ax
    """
    # Flatten the features and labels
    B, T, D = audio_feats.shape
    audio_feats_flat = audio_feats.reshape(-1, D)
    phoneme_feats_flat = phoneme_feats.reshape(-1, D)
    phoneme_labels_flat = phoneme_labels.reshape(-1)
    phone_scores_flat = phone_scores.reshape(-1)

    # Convert ignore_index to a set for efficient lookup
    if ignore_index is None:
        ignore_indices = set()
    elif isinstance(ignore_index, (list, tuple)):
        ignore_indices = set(ignore_index)
    else:
        ignore_indices = {ignore_index}
    
    # Filter out ignore_indices
    valid_indices = ~np.isin(phoneme_labels_flat, list(ignore_indices))
    audio_feats_valid = audio_feats_flat[valid_indices]
    phoneme_feats_valid = phoneme_feats_flat[valid_indices]
    phoneme_labels_valid = phoneme_labels_flat[valid_indices]
    phone_scores_valid = phone_scores_flat[valid_indices]

    # Get the most frequent phonemes
    unique, counts = np.unique(phoneme_labels_valid, return_counts=True)
    freq_phonemes = unique[np.argsort(-counts)][:max_phones]

    # Create a color palette
    #palette = sns.color_palette("hsv", len(freq_phonemes))
    # color_map = {phn: palette[i] for i, phn in enumerate(freq_phonemes)}
    # make color map with matplotlib
    import pdb; pdb.set_trace()
    cmap = plt.get_cmap('viridis', len(freq_phonemes))
    
    color_map = {phn: cmap(i) for i, phn in enumerate(freq_phonemes)}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    for phn in freq_phonemes:
        # compress_phoneme feats to 2D use tsn-e if D > 2
        if phoneme_feats_valid.shape[1] > 2:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=0, perplexity=5)
            phoneme_feats_valid_2d = tsne.fit_transform(phoneme_feats_valid)
            audio_feats_valid_2d = tsne.fit_transform(audio_feats_valid)
        else:
            phoneme_feats_valid_2d = phoneme_feats_valid
            audio_feats_valid_2d = audio_feats_valid
            
        phn_indices = phoneme_labels_valid == phn
        ax.scatter(
            phoneme_feats_valid_2d[phn_indices, 0],
            phoneme_feats_valid_2d[phn_indices, 1],
            c=[color_map[phn]],
            marker='x',
            label=f'Phoneme {phn} (Phoneme Feats)',
            alpha=1.0
        )
        ax.scatter(
            audio_feats_valid_2d[phn_indices, 0],
            audio_feats_valid_2d[phn_indices, 1],
            c=[color_map[phn]],
            marker='o',
            label=f'Phoneme {phn} (Audio Feats)',
            alpha=1.0
        )

    ax.set_title('CLAP-like Audio/Phoneme Clustering')
    ax.set_xlabel('Feature Dimension 1')
    ax.set_ylabel('Feature Dimension 2')
    ax.legend()
    plt.grid(True)
    
    return fig, ax 

def plot_phone_cluster(phoneme_feats, phoneme_labels, ignore_index=None, max_phones=5, label_encoder=None):
    """
    Plots the clustering of phoneme features in a t-SNE 2D space with centroids.
    Shows all instances of phoneme features with their centroids marked with 'x'.
    
    Parameters:
    - phoneme_feats: [B, T, D]
    - phoneme_labels: [B, T]
    - ignore_index: int, list, or tuple of indices to ignore in phoneme_labels (default: None)
    - max_phones: int, maximum number of phonemes to plot
    - label_encoder: optional encoder to convert phoneme IDs to strings
    return fig, ax
    """
    from sklearn.manifold import TSNE
    
    # Flatten the features and labels
    B, T, D = phoneme_feats.shape
    phoneme_feats_flat = phoneme_feats.reshape(-1, D)
    phoneme_labels_flat = phoneme_labels.reshape(-1)

    # Convert ignore_index to a set for efficient lookup
    if ignore_index is None:
        ignore_indices = set()
    elif isinstance(ignore_index, (list, tuple)):
        ignore_indices = set(ignore_index)
    else:
        ignore_indices = {ignore_index}
    
    # Filter out ignore_indices
    valid_indices = ~np.isin(phoneme_labels_flat, list(ignore_indices))
    phoneme_feats_valid = phoneme_feats_flat[valid_indices]
    phoneme_labels_valid = phoneme_labels_flat[valid_indices]

    # Get the most frequent phonemes
    unique, counts = np.unique(phoneme_labels_valid, return_counts=True)
    freq_phonemes = unique[np.argsort(-counts)][:max_phones]

    # Create a color palette
    cmap = plt.get_cmap('hsv', len(freq_phonemes))
    color_map = {phn: cmap(i) for i, phn in enumerate(freq_phonemes)}

    # Compress to 2D if needed
    if phoneme_feats_valid.shape[1] > 2:
        print(f"Compressing {phoneme_feats_valid.shape[0]} features to 2D using t-SNE...")
        tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(phoneme_feats_valid) // 3))
        phoneme_feats_2d = tsne.fit_transform(phoneme_feats_valid)
    else:
        phoneme_feats_2d = phoneme_feats_valid

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for phn in freq_phonemes:
        phn_indices = phoneme_labels_valid == phn
        color = color_map[phn]
        
        # Get phoneme label string
        if label_encoder is not None:
            phn_str = label_encoder.decode_ndim([phn])[0]
            phn_label = f'{phn_str}'
        else:
            phn_label = f'{phn}'
        
        # Plot instances (dots)
        ax.scatter(
            phoneme_feats_2d[phn_indices, 0],
            phoneme_feats_2d[phn_indices, 1],
            c=[color],
            marker='o',
            s=60,
            alpha=1.0,
            label=f'{phn_label} (instances)',
            edgecolors='none'
        )
        
        # Calculate and plot centroid
        centroid = phoneme_feats_2d[phn_indices].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            c=[color],
            marker='x',
            s=300,
            label=f'{phn_label} (centroid)',
            zorder=10
        )
    
    ax.set_title('Phoneme Feature Clustering with Centroids', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Simplify legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax