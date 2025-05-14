import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
from datetime import datetime
import wandb
import os
import sys
sys.path.append("./src/")
from dataset import train_val_test_split, get_adj_matrix
from trainer import fit
from models.gnn import GNN
from dicts import index, fields
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# get file date and time
now = datetime.now()
date = now.strftime("%Y-%m-%d")
time = now.strftime("%H-%M-%S")

# output dir (only for kaggle)
output_dir = "/kaggle/working" 

# randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load config
with open("./configs/gnn.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    
# wandb login
with open ("./configs/wandb.yaml", "r") as f:
    wandb_api_key = yaml.load(f, Loader=yaml.FullLoader)["wandb_api_key"]
    wandb.login(key=wandb_api_key)

# exp
m = cfg["m"]
variants = cfg["variants"]
history = cfg["history"]
horizons = cfg["horizons"]
batch_sizes = cfg["batch_sizes"]
lrs = cfg["lrs"]
seeds = cfg["seeds"]
epochs = cfg["epochs"]
hidden_dims = cfg["params"]["hidden_dims"]
n_layers = cfg["params"]["n_layers"]

variants_str = "-".join(str(item) for item in variants)
datasets_str = "-".join(str(item["name"]) for item in cfg["datasets"])
seeds_str = "-".join(str(item) for item in seeds)

# results
df_results = pd.DataFrame(columns=index+fields)
df_results = df_results.set_index(index)

# device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# for each batch size
for batch_size in batch_sizes:

    # for each seed
    for seed in seeds:

        # for each learning rate
        for lr in lrs:

            # for each horizon
            for h_idx, horizon in enumerate(horizons):

                # for each dataset
                for d in cfg["datasets"]:
                    torch.cuda.empty_cache()
                    
                    # params
                    d_embedding = d["params"]["d_embedding"]
                    if len(horizons) == len(cfg["params"]["hidden_dims"]):
                        hidden_dims = cfg["params"]["hidden_dims"][h_idx]
                    else:
                        hidden_dims = cfg["params"]["hidden_dims"][0] # default
                    epochs = cfg["epochs"]
                    if "epochs" in d:
                        epochs = d["epochs"]
                    
                    # load data
                    data = np.load(d["file"])
                    if "take_n" in d:
                        data = data[:d["take_n"]]
                        
                    # get adjacency matrix
                    df = pd.read_csv(d["file"].replace(".npy", ".csv"))
                    A, distA, fdistA, _ = get_adj_matrix(df)
                    
                    # datasets
                    n_steps, n_channels = data.shape
                    f_train, f_val_, f_test = d["split"]
                    split_str = "".join(str(int(item*10)) for item in d["split"])
                    dataset_train, dataset_val, dataset_test, embedding = train_val_test_split(
                        data, [f_train, f_test], history, horizon, d_embedding=d_embedding
                    )
                    
                    print("Batch size: {}, seed: {}, lr: {}, horizon: {}".format(batch_size, seed, lr, horizon))
                    print("{} ({}, {}, {}, {}), Hidden dims={}, Layers={}\r\n-----".format(
                        d["name"], len(data), len(dataset_train), len(dataset_val), len(dataset_test), 
                        hidden_dims, n_layers
                    ))

                    # for each variant
                    for v in variants:
                        # wandb init 
                        run_name = f"{d['name']}_{v}_h{horizon}_bs{batch_size}_lr{lr}_seed{seed}"
                        wandb.init(
                            project="time-series-benchmark-gnn",
                            name=run_name,
                            config={
                                "dataset": d["name"],
                                "model": m,
                                "variant": v,
                                "seed": seed,
                                "batch_size": batch_size,
                                "learning_rate": lr,
                                "epochs": epochs,
                                "history": history,
                                "horizon": horizon,
                                "hidden_dims": hidden_dims,
                                "n_layers": n_layers,
                                "d_embedding": d_embedding,
                                "split": d["split"],
                                "random_state": seed
                            },
                            reinit=True
                        )

                        # random state
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)

                        # dataloaders
                        dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4)
                        dataloader_val = DataLoader(dataset_val, batch_size, shuffle=True, num_workers=4)
                        dataloader_test = DataLoader(dataset_test, batch_size, shuffle=False, num_workers=4)
                        dataloaders = [dataloader_train, dataloader_val, dataloader_test]

                        # model initialization
                        # Note: You'll need to adapt the input_dim based on your data
                        # and how you construct the graph adjacency matrix
                        input_dim = history  # Adjust based on your needs
                        output_dim = horizon  # Predicting horizon steps ahead
                        
                        model = GNN(
                            input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            output_dim=output_dim,
                            n_layers=n_layers,
                            adj_matrix=A,
                            device=device
                        )

                        optimizer = Adam(model.parameters(), lr=lr)
                        loss_fn = nn.MSELoss()
                        metric_fn = nn.L1Loss()

                        # fit
                        # TODO: adapt the fit function to handle GNN inputs
                        # which typically need both node features and adjacency matrix
                        _, history_model = fit(model, optimizer, loss_fn, metric_fn, epochs, *dataloaders, desc=m + "/" + v)

                        # log
                        run = []
                        for epoch_id, metrics in enumerate(history_model):
                            run.append([
                                d["name"],      # dataset
                                n_steps,        # size
                                split_str,      # split
                                epochs,         # epochs
                                history,        # history
                                horizon,        # horizon
                                m,              # model
                                v,              # variant
                                seed,           # seed
                                batch_size,     # batch_size
                                lr,             # lr
                                hidden_dims,    # hidden_dims
                                n_layers,       # n_layers
                                d_embedding,    # d_embedding        
                                
                                epoch_id,       # epoch id
                                metrics[0,0],   # mse_train
                                metrics[0,1],   # mse_val
                                metrics[0,2],   # mse_test
                                metrics[1,0],   # mae_train
                                metrics[1,1],   # mae_val
                                metrics[1,2],   # mae_test
                            ])
                            
                            # wandb logging
                            wandb.log({
                                "mse_train":   metrics[0,0],
                                "mse_val":     metrics[0,1],
                                "mse_test":    metrics[0,2],
                                "mae_train":   metrics[1,0],
                                "mae_val":     metrics[1,1],
                                "mae_test":    metrics[1,2]
                            })

                        df_run = pd.DataFrame(run, columns=index+fields)
                        df_run = df_run.set_index(index)
                        df_results = pd.concat([df_results, df_run])

                        # save logs and checkpoints
                        folder = "{}_{}_GNN_{}_d_{}_e_{}_s_{}".format(
                            date, time, variants_str, datasets_str, epochs, seeds_str
                        )
                        filename = "{}-{}_GNN_{}_s-{}_{}_{}_{}_e-{}_L-{}_H-{}_bs-{}_lr-{}_hd-{}_nl-{}_emb-{}.pt".format(
                            date,
                            time,
                            v,
                            seed,
                            d["name"],
                            n_steps,
                            split_str,
                            epochs,
                            history,
                            horizon,
                            batch_size,
                            lr,
                            "-".join(map(str, hidden_dims)),
                            n_layers,
                            d_embedding,
                        )
                        # save_dir = os.path.join(output_dir, folder)
                        # os.makedirs(save_dir, exist_ok=True)
                        # model_path = os.path.join(save_dir, filename)
                        # torch.save(model.state_dict(), model_path)
                        
                        # # wandb artifact for model state
                        # artifact = wandb.Artifact(
                        #     name=filename.replace(".pt", ""),
                        #     type="model",
                        #     description=f"Model checkpoint for {v} variant, dataset {d['name']}, seed {seed}, horizon {horizon}"
                        # )
                        # artifact.add(model_path, name="model.pt")
                        # wandb.log_artifact(artifact)
                    
                    print("\r")
wandb.finish()