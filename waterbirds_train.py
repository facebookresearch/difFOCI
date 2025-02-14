# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#!/usr/bin/env python

import os
import sys
import json
import time
import torch
import submitit
import argparse
import numpy as np
import itertools

import difFOCI.models as models
from data.data_loading import get_loaders

def get_job_id():
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        return os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    if "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"]
    return None


class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def randl(l_):
    return l_[torch.randperm(len(l_))[0]]


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument('--output_dir', type=str, default='/checkpoint/krunolp/difFOCI')
    parser.add_argument('--data_path', type=str, default='/private/home/krunolp/foci_clean/difFOCI/data')
    parser.add_argument('--slurm_output_dir', type=str, default='slurm_outputs')
    parser.add_argument('--slurm_partition', type=str, default='learnlab')
    parser.add_argument('--max_time', type=int, default=1*3*60)
    parser.add_argument('--num_hparams_seeds', type=int, default=20)
    parser.add_argument('--num_init_seeds', type=int, default=5)
    parser.add_argument('--selector', type=str, default='min_acc_va')
    return vars(parser.parse_args())


def run_experiment(lr, lamda, bs, wd, seed, args):
    args["lr"] = lr
    args["lamda"] = lamda
    args["batch_size"] = bs
    args["init_seed"] = seed
    args["weight_decay"] = wd   
    
    # fixed
    args["beta"] = 0.2
    args["gamma"] = 1.
    
    start_time = time.time()
    torch.manual_seed(args["init_seed"])
    np.random.seed(args["init_seed"])
    loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], args["method"])
    job_id = get_job_id()

    sys.stdout = Tee(os.path.join(
        args["output_dir"], 
        'seed_{}_{}_{}.out'.format(job_id, args["hparams_seed"], args["init_seed"])), sys.stdout)
    sys.stderr = Tee(os.path.join(
        args["output_dir"], 
        'seed_{}_{}_{}.err'.format(job_id, args["hparams_seed"], args["init_seed"])), sys.stderr)
    checkpoint_file = os.path.join(
        args["output_dir"],
        'seed_{}_{}_{}.pt'.format(job_id, args["hparams_seed"], args["init_seed"]))
    best_checkpoint_file = os.path.join(
        args["output_dir"],
        "seed_{}_{}_{}.best.pt".format(job_id, args["hparams_seed"], args["init_seed"]),
    )


    model = {
        "erm": models.ERM,
        "dro": models.GroupDRO,
    }[args["method"]](args, loaders["tr"])

    last_epoch = 0
    best_selec_val = float('-inf')
    if os.path.exists(checkpoint_file):
        model.load(checkpoint_file)
        last_epoch = model.last_epoch
        best_selec_val = model.best_selec_val
        

    for epoch in range(last_epoch, args["num_epochs"]):
        for i, x, y, g in loaders["tr"]:
            model.update(i, x, y, g, epoch)

        result = {
            "args": args, "epoch": epoch, "time": time.time() - start_time}
        for loader_name, loader in loaders.items():
            avg_acc, group_accs = model.accuracy(loader)
            result["acc_" + loader_name] = group_accs
            result["avg_acc_" + loader_name] = avg_acc

        selec_value = {
            "min_acc_va": min(result["acc_va"]),
            "avg_acc_va": result["avg_acc_va"],
        }[args["selector"]]
        
        
        if selec_value >= best_selec_val:
            model.best_selec_val = selec_value
            best_selec_val = selec_value
            model.save(best_checkpoint_file)

        model.save(checkpoint_file)
        print(json.dumps(result))


if __name__ == "__main__":
    args = parse_args()
    seed = 12345
    
    torch.manual_seed(seed)
    args["hparams_seed"] = seed
    args["init_seed"] = seed
    
    args["dataset"] =  "waterbirds"
    args["method"] = "erm"
    args["num_epochs"] = 300 + 60
    args["eta"] = 0.1


    os.makedirs(args["output_dir"], exist_ok=True)
    torch.manual_seed(0)
    
    
    executor = submitit.AutoExecutor(folder=args['slurm_output_dir'])
    executor.update_parameters(
        slurm_time=args["max_time"],
        gpus_per_node=1,
        slurm_array_parallelism=512,
        cpus_per_task=4,
        slurm_partition=args["slurm_partition"])
    
    lrs = [1e-2, 1e-3, 1e-4]
    lamdas = [1e-2, 1e-3, 1e-4]
    wds = [1e-2, 1e-3, 1e-4]
    batch_sizes = [4, 8, 16, 32]    
        
    
    jobs = []
    with executor.batch():
        for lr, lamda, bs, wd in list(itertools.product(lrs, lamdas, batch_sizes, wds)):
            job = executor.submit(run_experiment, lr, lamda, bs, wd, seed, args)
            jobs.append(job)
