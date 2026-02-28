import wandb

def wandb_log(wandb_run, rank, log_item, commit=True):
    if rank == 0:
        wandb_run.log(log_item, commit=commit)