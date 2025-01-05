# utils/model_tracker.py

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import re
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class ModelTracker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.readme_path = project_root / "README.md"
        self.metrics_path = project_root / "results" / "metrics"
        self.metrics_path.mkdir(parents=True, exist_ok=True)

    def save_model_metrics(self, 
                         model_name: str, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         hyperparameters: dict = None, 
                         notes: str = None,
                         fold: str = "test"):
        """
        Save model metrics and update README.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        hyperparameters : dict, optional
            Model hyperparameters
        notes : str, optional
            Additional notes about the model run
        fold : str, optional
            Specify if metrics are for 'train', 'val', or 'test' set
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "timestamp": timestamp,
            "fold": fold,
            "r2_score": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "hyperparameters": hyperparameters,
            "notes": notes
        }

        # Save metrics
        metrics_file = self.metrics_path / f"{model_name}_{timestamp}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # Update README
        self.update_readme()

    def update_readme(self):
        """Update README with latest model comparisons."""
        # Get all metrics files
        metrics_files = list(self.metrics_path.glob("*_metrics.json"))
        all_metrics = []

        for file in metrics_files:
            with open(file, "r") as f:
                metrics = json.load(f)
                all_metrics.append(metrics)

        # Sort by timestamp (most recent first)
        all_metrics.sort(key=lambda x: x["timestamp"], reverse=True)

        # Create README content
        readme_content = self._generate_readme_content(all_metrics)

        # Update README while preserving non-results sections
        if self.readme_path.exists():
            with open(self.readme_path, "r") as f:
                current_content = f.read()
            
            # Find the results section and replace it
            pattern = r"(## Model Results.*?)(?=##|\Z)"
            if re.search(pattern, current_content, re.DOTALL):
                new_content = re.sub(pattern, readme_content, current_content, flags=re.DOTALL)
            else:
                new_content = current_content + "\n" + readme_content
        else:
            new_content = readme_content

        with open(self.readme_path, "w") as f:
            f.write(new_content)

    def _generate_readme_content(self, all_metrics: list) -> str:
        """Generate formatted README content."""
        content = [
            "## Model Results\n",
            "### Performance Comparison\n",
            "| Model | Date | Fold | R² Score | RMSE | MAE | Notes |",
            "|-------|------|------|----------|------|-----|--------|"
        ]

        for metrics in all_metrics:
            date = datetime.strptime(metrics["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d")
            row = [
                metrics["model_name"],
                date,
                metrics["fold"],
                f"{metrics['r2_score']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                metrics.get("notes", "")
            ]
            content.append("| " + " | ".join(row) + " |")

        content.extend([
            "\n### Hyperparameter Details\n",
            "| Model | Date | Parameters |",
            "|-------|------|------------|"
        ])

        for metrics in all_metrics:
            if metrics.get("hyperparameters"):
                date = datetime.strptime(metrics["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d")
                params = "<br>".join([f"{k}: {v}" for k, v in metrics["hyperparameters"].items()])
                content.append(f"| {metrics['model_name']} | {date} | {params} |")

        return "\n".join(content) + "\n"