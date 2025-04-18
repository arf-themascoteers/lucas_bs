from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"6_ad_cnn_vs_bsdr_cnn"
    tasks = {
        "algorithms": ["bsdrcnn_nosig","bsdrcnn"],
        "datasets": ["lucas"],
        "target_sizes": [8, 16, 32, 64, 128, 256, 512],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 10
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
