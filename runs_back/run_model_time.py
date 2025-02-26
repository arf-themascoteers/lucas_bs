from task_runner import TaskRunner
from my_env import TEST

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tasks = {
        "algorithms" : ["bsdrcnn","bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [8,16,32,64,128,256,512],
        "scale_y" : ["robust"],
        "mode" : ["dyn","static"],
        "train_sizes":train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,"model_time",verbose=verbose)
    ev.evaluate()
