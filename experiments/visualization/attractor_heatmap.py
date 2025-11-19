"""
Attractor Heatmap Visualization

Creates heatmaps showing:
- Which attractors activate for which tokens
- Match scores over time
- Attractor formation, reinforcement, decay patterns

Used for Phase 5 Task 3: Instrumentation & Visualization

Author: Halcyon AI Research
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import json


class AttractorHeatmapVisualizer:
    """
    Visualize attractor dynamics as heatmaps.

    Example:
        >>> visualizer = AttractorHeatmapVisualizer()
        >>> logs = get_drai_logs(model)  # From enabled logging
        >>> visualizer.plot_activation_heatmap(logs, tokens)
        >>> visualizer.save("results/phase5/heatmap.png")
    """

    def __init__(self, figsize=(14, 8), dpi=300):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size (width, height)
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

        # Color schemes
        self.cmap_activation = 'YlOrRd'  # Yellow to Orange to Red
        self.cmap_actions = self._create_action_colormap()

    def _create_action_colormap(self):
        """Create custom colormap for attractor actions."""
        colors = [
            (0.9, 0.9, 0.9),  # None/neutral - light gray
            (0.6, 0.8, 1.0),  # Create - light blue
            (1.0, 0.8, 0.6),  # Reinforce - light orange
            (1.0, 0.6, 0.6),  # Replace - light red
        ]
        return LinearSegmentedColormap.from_list('actions', colors, N=4)

    def plot_activation_heatmap(
        self,
        logs: List[Dict],
        tokens: Optional[List[str]] = None,
        max_attractors: int = 32,
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of attractor activations over time.

        Args:
            logs: List of log entries from DRAI layer
            tokens: Optional list of token strings for x-axis labels
            max_attractors: Maximum number of attractors to show
            save_path: Optional path to save figure

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Extract data
        num_steps = len(logs)
        match_scores = []  # [steps, attractors]
        attractor_counts = []
        actions = []

        for log in logs:
            scores = log.get('match_scores', [])
            # Pad to max_attractors
            padded = scores + [0] * (max_attractors - len(scores))
            match_scores.append(padded[:max_attractors])

            attractor_counts.append(log.get('attractor_count', 0))
            actions.append(log.get('action', 'none'))

        match_scores = np.array(match_scores)  # [steps, attractors]

        # Create figure
        fig, (ax_heat, ax_count, ax_action) = plt.subplots(
            3, 1,
            figsize=self.figsize,
            height_ratios=[5, 1, 0.5],
            gridspec_kw={'hspace': 0.3}
        )

        # Main heatmap
        im = ax_heat.imshow(
            match_scores.T,  # Transpose: attractors on y-axis, steps on x-axis
            aspect='auto',
            cmap=self.cmap_activation,
            interpolation='nearest',
            vmin=0,
            vmax=1
        )

        # Labels
        ax_heat.set_xlabel('Step (Token Position)', fontsize=12)
        ax_heat.set_ylabel('Attractor Index', fontsize=12)
        ax_heat.set_title('Attractor Activation Heatmap\n(Match Scores Over Time)', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Match Score (Cosine Similarity)', fontsize=10)

        # Add token labels if provided
        if tokens and len(tokens) == num_steps:
            # Show every N tokens to avoid crowding
            step_every = max(1, len(tokens) // 20)
            tick_positions = range(0, len(tokens), step_every)
            tick_labels = [tokens[i] if i < len(tokens) else '' for i in tick_positions]

            ax_heat.set_xticks(tick_positions)
            ax_heat.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

        # Attractor count subplot
        ax_count.plot(range(num_steps), attractor_counts, color='steelblue', linewidth=2)
        ax_count.fill_between(range(num_steps), attractor_counts, alpha=0.3, color='steelblue')
        ax_count.set_ylabel('Count', fontsize=10)
        ax_count.set_title('Active Attractor Count', fontsize=11)
        ax_count.grid(True, alpha=0.3)
        ax_count.set_xlim(0, num_steps)

        # Action subplot (create, reinforce, replace, none)
        action_map = {'none': 0, 'create': 1, 'reinforce': 2, 'replace': 3}
        action_values = [action_map.get(a, 0) for a in actions]
        action_array = np.array(action_values).reshape(1, -1)

        im_action = ax_action.imshow(
            action_array,
            aspect='auto',
            cmap=self.cmap_actions,
            interpolation='nearest',
            vmin=0,
            vmax=3
        )

        ax_action.set_yticks([])
        ax_action.set_xlabel('Step', fontsize=10)
        ax_action.set_title('Attractor Actions', fontsize=11)

        # Legend for actions
        legend_elements = [
            mpatches.Patch(color=(0.9, 0.9, 0.9), label='None'),
            mpatches.Patch(color=(0.6, 0.8, 1.0), label='Create'),
            mpatches.Patch(color=(1.0, 0.8, 0.6), label='Reinforce'),
            mpatches.Patch(color=(1.0, 0.6, 0.6), label='Replace')
        ]
        ax_action.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        return fig, (ax_heat, ax_count, ax_action)

    def plot_attractor_strength_heatmap(
        self,
        logs: List[Dict],
        max_attractors: int = 32,
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of attractor strengths (coherence) over time.

        Args:
            logs: List of log entries from DRAI layer
            max_attractors: Maximum number of attractors to show
            save_path: Optional path to save figure

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Extract attractor strengths
        num_steps = len(logs)
        strengths = []

        for log in logs:
            strength_list = log.get('attractor_strengths', [])
            # Pad to max_attractors
            padded = strength_list + [0] * (max_attractors - len(strength_list))
            strengths.append(padded[:max_attractors])

        strengths = np.array(strengths).T  # [attractors, steps]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Heatmap
        im = ax.imshow(
            strengths,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )

        # Labels
        ax.set_xlabel('Step (Token Position)', fontsize=12)
        ax.set_ylabel('Attractor Index', fontsize=12)
        ax.set_title('Attractor Strength (Coherence) Over Time', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Coherence', fontsize=10)

        # Add grid for readability
        ax.grid(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Strength heatmap saved to {save_path}")

        return fig, ax

    def plot_entropy_over_time(
        self,
        logs: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot attractor entropy over time.

        Args:
            logs: List of log entries
            save_path: Optional path to save figure

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Extract entropy
        entropies = [log.get('attractor_entropy', 0) for log in logs]
        steps = range(len(entropies))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot
        ax.plot(steps, entropies, color='purple', linewidth=2, alpha=0.8)
        ax.fill_between(steps, entropies, alpha=0.2, color='purple')

        # Labels
        ax.set_xlabel('Step (Token Position)', fontsize=12)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title('Attractor Distribution Entropy Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add interpretation
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No uncertainty')
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Entropy plot saved to {save_path}")

        return fig, ax

    def plot_novelty_vs_match(
        self,
        logs: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot proportion of novel vs matched patterns over time.

        Args:
            logs: List of log entries
            save_path: Optional path to save figure

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Extract novelty info
        novel_count = sum(1 for log in logs if log.get('is_novel', False))
        matched_count = len(logs) - novel_count

        # Cumulative over time
        cumulative_novel = []
        cumulative_matched = []
        running_novel = 0
        running_matched = 0

        for log in logs:
            if log.get('is_novel', False):
                running_novel += 1
            else:
                running_matched += 1

            cumulative_novel.append(running_novel)
            cumulative_matched.append(running_matched)

        steps = range(len(logs))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Cumulative counts
        ax1.plot(steps, cumulative_novel, label='Novel', color='orange', linewidth=2)
        ax1.plot(steps, cumulative_matched, label='Matched', color='green', linewidth=2)
        ax1.fill_between(steps, cumulative_novel, alpha=0.2, color='orange')
        ax1.fill_between(steps, cumulative_matched, alpha=0.2, color='green')

        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Cumulative Count', fontsize=12)
        ax1.set_title('Novel vs Matched Patterns (Cumulative)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right: Pie chart
        colors = ['orange', 'green']
        ax2.pie(
            [novel_count, matched_count],
            labels=['Novel', 'Matched'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax2.set_title(f'Overall Distribution\n(Total: {len(logs)} steps)', fontsize=13, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Novelty plot saved to {save_path}")

        return fig, (ax1, ax2)

    def create_summary_figure(
        self,
        logs: List[Dict],
        tokens: Optional[List[str]] = None,
        story_text: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive summary figure with multiple subplots.

        Args:
            logs: List of log entries
            tokens: Optional token list
            story_text: Optional story text for context
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        if not logs:
            print("No logs to visualize")
            return None

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

        # 1. Main activation heatmap (large, top)
        ax_heat = fig.add_subplot(gs[0:2, :])

        num_steps = len(logs)
        match_scores = []
        for log in logs:
            scores = log.get('match_scores', [])
            match_scores.append(scores + [0] * (32 - len(scores)))
        match_scores = np.array(match_scores).T

        im = ax_heat.imshow(match_scores, aspect='auto', cmap=self.cmap_activation, vmin=0, vmax=1)
        ax_heat.set_xlabel('Step', fontsize=11)
        ax_heat.set_ylabel('Attractor Index', fontsize=11)
        ax_heat.set_title('Attractor Activation Heatmap', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label='Match Score')

        # 2. Attractor count (bottom left)
        ax_count = fig.add_subplot(gs[2, 0])
        counts = [log.get('attractor_count', 0) for log in logs]
        ax_count.plot(range(num_steps), counts, color='steelblue', linewidth=2)
        ax_count.fill_between(range(num_steps), counts, alpha=0.3, color='steelblue')
        ax_count.set_xlabel('Step', fontsize=10)
        ax_count.set_ylabel('Count', fontsize=10)
        ax_count.set_title('Active Attractors', fontsize=11)
        ax_count.grid(True, alpha=0.3)

        # 3. Entropy (bottom right)
        ax_entropy = fig.add_subplot(gs[2, 1])
        entropies = [log.get('attractor_entropy', 0) for log in logs]
        ax_entropy.plot(range(num_steps), entropies, color='purple', linewidth=2)
        ax_entropy.fill_between(range(num_steps), entropies, alpha=0.2, color='purple')
        ax_entropy.set_xlabel('Step', fontsize=10)
        ax_entropy.set_ylabel('Entropy (nats)', fontsize=10)
        ax_entropy.set_title('Attractor Entropy', fontsize=11)
        ax_entropy.grid(True, alpha=0.3)

        # 4. Actions (bottom left, second row)
        ax_actions = fig.add_subplot(gs[3, 0])
        action_map = {'none': 0, 'create': 1, 'reinforce': 2, 'replace': 3}
        actions = [action_map.get(log.get('action', 'none'), 0) for log in logs]
        action_counts = [actions.count(i) for i in range(4)]
        ax_actions.bar(['None', 'Create', 'Reinforce', 'Replace'], action_counts,
                      color=[(0.9,0.9,0.9), (0.6,0.8,1.0), (1.0,0.8,0.6), (1.0,0.6,0.6)])
        ax_actions.set_ylabel('Count', fontsize=10)
        ax_actions.set_title('Action Distribution', fontsize=11)
        ax_actions.grid(axis='y', alpha=0.3)

        # 5. Novel vs Matched (bottom right, second row)
        ax_novel = fig.add_subplot(gs[3, 1])
        novel_count = sum(1 for log in logs if log.get('is_novel', False))
        matched_count = num_steps - novel_count
        ax_novel.pie([novel_count, matched_count], labels=['Novel', 'Matched'],
                    colors=['orange', 'green'], autopct='%1.1f%%', startangle=90)
        ax_novel.set_title('Novel vs Matched', fontsize=11)

        # Overall title
        fig.suptitle('DRAI Attractor Dynamics Summary', fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Summary figure saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage with synthetic data
    print("="*70)
    print("ATTRACTOR HEATMAP VISUALIZATION - Example")
    print("="*70)

    # Generate synthetic log data for demonstration
    np.random.seed(42)
    num_steps = 50
    max_attractors = 10

    logs = []
    for step in range(num_steps):
        # Simulate attractor dynamics
        num_active = min(step // 5 + 1, max_attractors)  # Gradually add attractors

        match_scores = np.random.beta(2, 5, num_active).tolist()  # Skewed toward lower values
        match_scores.sort(reverse=True)  # Strongest first

        attractor_strengths = np.random.beta(3, 2, num_active).tolist()  # Skewed toward higher values

        action = np.random.choice(['none', 'create', 'reinforce', 'replace'],
                                  p=[0.5, 0.1, 0.35, 0.05])

        log_entry = {
            'step': step,
            'attractor_count': num_active,
            'match_scores': match_scores,
            'attractor_strengths': attractor_strengths,
            'action': action,
            'attractor_entropy': np.random.uniform(0, 2),
            'is_novel': np.random.random() < 0.3,
            'best_match_idx': 0 if match_scores else -1,
            'best_match_sim': match_scores[0] if match_scores else 0.0
        }
        logs.append(log_entry)

    # Create visualizations
    visualizer = AttractorHeatmapVisualizer()

    Path("results/phase5/visualizations").mkdir(parents=True, exist_ok=True)

    # 1. Activation heatmap
    print("\nGenerating activation heatmap...")
    visualizer.plot_activation_heatmap(
        logs,
        save_path="results/phase5/visualizations/activation_heatmap.png"
    )

    # 2. Strength heatmap
    print("Generating strength heatmap...")
    visualizer.plot_attractor_strength_heatmap(
        logs,
        save_path="results/phase5/visualizations/strength_heatmap.png"
    )

    # 3. Entropy plot
    print("Generating entropy plot...")
    visualizer.plot_entropy_over_time(
        logs,
        save_path="results/phase5/visualizations/entropy_plot.png"
    )

    # 4. Novelty analysis
    print("Generating novelty analysis...")
    visualizer.plot_novelty_vs_match(
        logs,
        save_path="results/phase5/visualizations/novelty_analysis.png"
    )

    # 5. Summary figure
    print("Generating summary figure...")
    visualizer.create_summary_figure(
        logs,
        save_path="results/phase5/visualizations/summary.png"
    )

    print("\n" + "="*70)
    print("Visualizations complete!")
    print("Saved to: results/phase5/visualizations/")
    print("="*70)
