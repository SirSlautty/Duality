"""
Attractor Time-Series Visualization

Creates time-series plots showing:
- Attractor strength evolution over time
- Top-k attractor tracking
- Formation, reinforcement, decay patterns
- Correlation with story events

Used for Phase 5 Task 3: Instrumentation & Visualization

Author: Halcyon AI Research
Date: 2025-11-18
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class AttractorTimeSeriesVisualizer:
    """
    Visualize attractor dynamics as time-series.

    Example:
        >>> visualizer = AttractorTimeSeriesVisualizer()
        >>> logs = get_drai_logs(model)
        >>> visualizer.plot_top_k_attractors(logs, k=5)
        >>> visualizer.save("results/phase5/time_series.png")
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

        # Color palette for multiple attractors
        self.colors = sns.color_palette("husl", 10)

    def plot_top_k_attractors(
        self,
        logs: List[Dict],
        k: int = 5,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot strength evolution of top-k most active attractors.

        Args:
            logs: List of log entries from DRAI layer
            k: Number of top attractors to track
            tokens: Optional token list for context
            save_path: Optional path to save figure

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Extract attractor strengths over time
        num_steps = len(logs)
        max_attractors = max(log.get('attractor_count', 0) for log in logs)

        # Build full strength matrix [attractors, steps]
        strengths = np.zeros((max_attractors, num_steps))

        for step, log in enumerate(logs):
            strength_list = log.get('attractor_strengths', [])
            for idx, strength in enumerate(strength_list):
                if idx < max_attractors:
                    strengths[idx, step] = strength

        # Find top-k attractors by mean strength
        mean_strengths = strengths.mean(axis=1)
        top_k_indices = np.argsort(mean_strengths)[-k:][::-1]  # Top k, descending

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each top attractor
        for i, attractor_idx in enumerate(top_k_indices):
            color = self.colors[i % len(self.colors)]
            strength_trace = strengths[attractor_idx, :]

            ax.plot(
                range(num_steps),
                strength_trace,
                label=f'Attractor {attractor_idx} (μ={mean_strengths[attractor_idx]:.3f})',
                color=color,
                linewidth=2.5,
                alpha=0.8
            )

            # Fill area for emphasis
            ax.fill_between(
                range(num_steps),
                strength_trace,
                alpha=0.15,
                color=color
            )

        # Labels and formatting
        ax.set_xlabel('Step (Token Position)', fontsize=12)
        ax.set_ylabel('Attractor Strength (Coherence)', fontsize=12)
        ax.set_title(f'Top-{k} Attractor Strength Evolution Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, num_steps)
        ax.set_ylim(0, strengths.max() * 1.1)

        # Add formation/decay annotations
        self._annotate_lifecycle_events(ax, logs, top_k_indices[:3])  # Top 3 only

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Time-series saved to {save_path}")

        return fig, ax

    def _annotate_lifecycle_events(
        self,
        ax,
        logs: List[Dict],
        attractor_indices: List[int]
    ):
        """Add annotations for attractor formation/decay events."""
        for attractor_idx in attractor_indices:
            # Find first non-zero (formation)
            formation_step = None
            for step, log in enumerate(logs):
                strengths = log.get('attractor_strengths', [])
                if attractor_idx < len(strengths) and strengths[attractor_idx] > 0.1:
                    formation_step = step
                    break

            if formation_step is not None:
                ax.axvline(formation_step, color='green', linestyle=':', alpha=0.3, linewidth=1)

    def plot_attractor_lifecycle(
        self,
        logs: List[Dict],
        attractor_idx: int,
        tokens: Optional[List[str]] = None,
        events: Optional[List[Tuple[int, str]]] = None,  # [(step, "Alice mentioned"), ...]
        save_path: Optional[str] = None
    ):
        """
        Plot detailed lifecycle of a single attractor with context.

        Args:
            logs: List of log entries
            attractor_idx: Which attractor to track
            tokens: Optional token list
            events: Optional list of (step, event_description) tuples for annotations
            save_path: Optional path to save

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Extract data for this attractor
        num_steps = len(logs)
        strengths = []
        match_scores = []
        actions = []

        for log in logs:
            strength_list = log.get('attractor_strengths', [])
            if attractor_idx < len(strength_list):
                strengths.append(strength_list[attractor_idx])
            else:
                strengths.append(0.0)

            scores = log.get('match_scores', [])
            if attractor_idx < len(scores):
                match_scores.append(scores[attractor_idx])
            else:
                match_scores.append(0.0)

            # Track if this attractor was involved in action
            if log.get('best_match_idx') == attractor_idx:
                actions.append(log.get('action', 'none'))
            else:
                actions.append('none')

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(self.figsize[0], self.figsize[1] + 2),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 2, 1], 'hspace': 0.15}
        )

        steps = range(num_steps)

        # Subplot 1: Strength
        ax1.plot(steps, strengths, color='steelblue', linewidth=3, label='Strength (Coherence)')
        ax1.fill_between(steps, strengths, alpha=0.3, color='steelblue')
        ax1.set_ylabel('Strength', fontsize=11)
        ax1.set_title(f'Attractor {attractor_idx} Lifecycle', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Add event annotations if provided
        if events:
            for event_step, event_label in events:
                ax1.axvline(event_step, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax1.text(event_step, ax1.get_ylim()[1] * 0.9, event_label,
                        rotation=90, va='top', ha='right', fontsize=8, color='red')

        # Subplot 2: Match scores
        ax2.plot(steps, match_scores, color='orange', linewidth=2, label='Match Score')
        ax2.fill_between(steps, match_scores, alpha=0.3, color='orange')
        ax2.set_ylabel('Match Score', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Actions
        action_map = {'none': 0, 'create': 1, 'reinforce': 2, 'replace': 3}
        action_values = [action_map.get(a, 0) for a in actions]

        colors_action = [(0.9,0.9,0.9) if v == 0 else
                        (0.6,0.8,1.0) if v == 1 else
                        (1.0,0.8,0.6) if v == 2 else
                        (1.0,0.6,0.6) for v in action_values]

        ax3.bar(steps, action_values, color=colors_action, width=1.0, edgecolor='none')
        ax3.set_xlabel('Step (Token Position)', fontsize=11)
        ax3.set_ylabel('Action', fontsize=11)
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['None', 'Create', 'Reinforce', 'Replace'], fontsize=9)
        ax3.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Lifecycle plot saved to {save_path}")

        return fig, (ax1, ax2, ax3)

    def plot_attractor_correlation(
        self,
        logs: List[Dict],
        max_attractors: int = 32,
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrix between attractors over time.

        Args:
            logs: List of log entries
            max_attractors: Maximum attractors to include
            save_path: Optional path to save

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Build strength matrix [attractors, steps]
        num_steps = len(logs)
        strengths = np.zeros((max_attractors, num_steps))

        for step, log in enumerate(logs):
            strength_list = log.get('attractor_strengths', [])
            for idx, strength in enumerate(strength_list):
                if idx < max_attractors:
                    strengths[idx, step] = strength

        # Compute correlation matrix
        # Only use attractors that were active at some point
        active_mask = strengths.max(axis=1) > 0.1
        active_strengths = strengths[active_mask, :]
        num_active = active_strengths.shape[0]

        if num_active < 2:
            print("Not enough active attractors for correlation analysis")
            return None, None

        correlation = np.corrcoef(active_strengths)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Heatmap
        im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        # Labels
        active_indices = np.where(active_mask)[0]
        ax.set_xticks(range(num_active))
        ax.set_yticks(range(num_active))
        ax.set_xticklabels([f'A{i}' for i in active_indices], fontsize=9)
        ax.set_yticklabels([f'A{i}' for i in active_indices], fontsize=9)

        ax.set_xlabel('Attractor Index', fontsize=11)
        ax.set_ylabel('Attractor Index', fontsize=11)
        ax.set_title('Attractor Correlation Matrix\n(Temporal Co-occurrence)', fontsize=13, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson Correlation', fontsize=10)

        # Add correlation values as text
        for i in range(num_active):
            for j in range(num_active):
                text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(correlation[i, j]) < 0.5 else "white",
                             fontsize=7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")

        return fig, ax

    def plot_formation_decay_timeline(
        self,
        logs: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot timeline showing when attractors form and decay.

        Args:
            logs: List of log entries
            save_path: Optional path to save

        Returns:
            Figure and axes objects
        """
        if not logs:
            print("No logs to visualize")
            return None, None

        # Track attractor lifespans
        attractor_lifespans = {}  # {attractor_idx: {'birth': step, 'death': step or None}}

        num_steps = len(logs)

        for step, log in enumerate(logs):
            strengths = log.get('attractor_strengths', [])

            for attractor_idx, strength in enumerate(strengths):
                if attractor_idx not in attractor_lifespans and strength > 0.1:
                    # Birth
                    attractor_lifespans[attractor_idx] = {'birth': step, 'death': None}

                elif attractor_idx in attractor_lifespans and attractor_lifespans[attractor_idx]['death'] is None:
                    if strength < 0.05:
                        # Death
                        attractor_lifespans[attractor_idx]['death'] = step

        # Close any still-alive attractors
        for attractor_idx in attractor_lifespans:
            if attractor_lifespans[attractor_idx]['death'] is None:
                attractor_lifespans[attractor_idx]['death'] = num_steps

        # Create figure
        fig, ax = plt.subplots(figsize=(self.figsize[0], len(attractor_lifespans) * 0.5 + 2))

        # Plot each attractor lifespan as a horizontal bar
        for i, (attractor_idx, lifespan) in enumerate(sorted(attractor_lifespans.items())):
            birth = lifespan['birth']
            death = lifespan['death']
            duration = death - birth

            ax.barh(i, duration, left=birth, height=0.8,
                   color=self.colors[attractor_idx % len(self.colors)],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=0.5)

            # Label
            ax.text(birth - 1, i, f'A{attractor_idx}',
                   va='center', ha='right', fontsize=9, fontweight='bold')

        # Labels
        ax.set_xlabel('Step (Token Position)', fontsize=12)
        ax.set_ylabel('Attractor', fontsize=12)
        ax.set_title('Attractor Formation & Decay Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(attractor_lifespans)))
        ax.set_yticklabels(['' for _ in attractor_lifespans])  # Labels already on left
        ax.set_xlim(0, num_steps)
        ax.grid(axis='x', alpha=0.3)

        # Legend
        ax.axvline(0, color='green', linestyle='--', alpha=0.5, label='Birth', linewidth=2)
        ax.axvline(num_steps, color='red', linestyle='--', alpha=0.5, label='Death', linewidth=2)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Timeline saved to {save_path}")

        return fig, ax


if __name__ == "__main__":
    # Example usage with synthetic data
    print("="*70)
    print("ATTRACTOR TIME-SERIES VISUALIZATION - Example")
    print("="*70)

    # Generate synthetic log data
    np.random.seed(42)
    num_steps = 100
    max_attractors = 8

    logs = []
    for step in range(num_steps):
        # Simulate growing and decaying attractors
        num_active = min((step // 10) + 1, max_attractors)

        strengths = []
        for i in range(num_active):
            # Some attractors grow, some decay
            if i < 3:  # Persistent attractors
                strength = min(0.3 + step * 0.007, 1.0)
            else:  # Transient attractors
                strength = 0.5 * np.exp(-(step - i*10)**2 / 100)

            strengths.append(max(0, strength + np.random.normal(0, 0.05)))

        match_scores = [s * np.random.uniform(0.8, 1.0) for s in strengths]

        best_idx = np.argmax(strengths) if strengths else -1
        action = np.random.choice(['none', 'create', 'reinforce'], p=[0.6, 0.1, 0.3])

        log_entry = {
            'step': step,
            'attractor_count': num_active,
            'attractor_strengths': strengths,
            'match_scores': match_scores,
            'best_match_idx': best_idx,
            'action': action,
        }
        logs.append(log_entry)

    # Create visualizations
    visualizer = AttractorTimeSeriesVisualizer()

    Path("results/phase5/visualizations").mkdir(parents=True, exist_ok=True)

    # 1. Top-k attractors
    print("\nGenerating top-k attractor plot...")
    visualizer.plot_top_k_attractors(
        logs,
        k=5,
        save_path="results/phase5/visualizations/top_k_attractors.png"
    )

    # 2. Single attractor lifecycle
    print("Generating lifecycle plot...")
    events = [(25, "Event A"), (50, "Event B"), (75, "Event C")]
    visualizer.plot_attractor_lifecycle(
        logs,
        attractor_idx=0,
        events=events,
        save_path="results/phase5/visualizations/attractor_lifecycle.png"
    )

    # 3. Correlation matrix
    print("Generating correlation matrix...")
    visualizer.plot_attractor_correlation(
        logs,
        save_path="results/phase5/visualizations/attractor_correlation.png"
    )

    # 4. Formation/decay timeline
    print("Generating formation/decay timeline...")
    visualizer.plot_formation_decay_timeline(
        logs,
        save_path="results/phase5/visualizations/formation_decay_timeline.png"
    )

    print("\n" + "="*70)
    print("Time-series visualizations complete!")
    print("Saved to: results/phase5/visualizations/")
    print("="*70)
