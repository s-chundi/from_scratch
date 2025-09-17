import wandb
from glob import glob
from fnmatch import fnmatch

def wandb_log(wandb_run, rank, log_item):
    if rank == 0:
        wandb_run.log(log_item)


def wandb_save_files(wandb_run):
    """
    Kind of a brute force way to use wandb as github repo.
    Github does not play nicely with my remote machine.
    """
    with open('.gitignore', 'r') as f:
        gitignore_lines = f.readlines()
    gitignore_patterns = [line.strip() for line in gitignore_lines if line.strip() and not line.strip().startswith('#')]
    for file in glob("**/*.py", recursive=True):   
        ignored = False
        for pattern in gitignore_patterns:
            if fnmatch(file, pattern):
                ignored = True
                break
        if not ignored:
            wandb_run.save(file)