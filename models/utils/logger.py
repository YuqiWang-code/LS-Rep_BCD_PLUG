"""
Training Logger for A2Net
"""

import os
import time


class TrainingLogger:
    def __init__(self, log_file, config):
        self.log_file = log_file
        self.config = config
        self.start_time = time.time()

        # Write header
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("A2Net Training Log\n")
            f.write("=" * 80 + "\n\n")

            f.write("Experiment Configuration:\n")
            f.write("-" * 80 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 80 + "\n\n")

            f.write("Training Progress:\n")
            f.write("-" * 80 + "\n")

    def log_epoch(self, epoch, total_epochs, train_losses, val_metrics, lr, gpu_mem, is_best):
        """Log epoch training and validation results"""
        log_str = f"Epoch [{epoch}/{total_epochs}] "
        log_str += f"Loss: {train_losses['total']:.4f} "
        log_str += f"F1: {val_metrics['f1']:.4f} "
        log_str += f"IoU: {val_metrics['iou']:.4f} "
        log_str += f"Kappa: {val_metrics['kappa']:.4f} "
        log_str += f"LR: {lr:.6f} "
        log_str += f"GPU: {gpu_mem:.2f}GB"

        if is_best:
            log_str += " ⭐ BEST"

        print(log_str)

        with open(self.log_file, 'a') as f:
            f.write(log_str + "\n")

    def log_test_results(self, test_metrics, model_file):
        """Log final test results"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Test Results (Best Model)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model File: {model_file}\n")
            f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"F1:        {test_metrics['f1']:.4f}\n")
            f.write(f"IoU:       {test_metrics['iou']:.4f}\n")
            f.write(f"OA:        {test_metrics['oa']:.4f}\n")
            f.write(f"Kappa:     {test_metrics['kappa']:.4f}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Training Time: {hours}h {minutes}m {seconds}s\n")
            f.write("=" * 80 + "\n")

    def log_message(self, message):
        """Log a custom message"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
