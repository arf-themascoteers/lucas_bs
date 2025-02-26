from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"ablation_fc"
    tasks = {
        "algorithms": ["bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [8,16,32,64,128,256,512,4200],
        "scale_y": ["robust"],
        "mode": ["static","dyn"],
        "train_sizes": train_sizes
    }
    folds = 10
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
