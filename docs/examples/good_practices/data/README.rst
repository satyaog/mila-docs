.. NOTE: This file is auto-generated from examples/good_practices/data/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

Data
====


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_
* `examples/distributed/single_gpu <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu>`_

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/data>`_


**job.sh**

.. code:: diff

    # distributed/single_gpu/job.sh -> good_practices/data/job.sh
    #!/bin/bash
    #SBATCH --gpus-per-task=rtx8000:1
    #SBATCH --cpus-per-task=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --mem=16G
   -#SBATCH --time=00:15:00
   +#SBATCH --time=01:30:00
   +set -o errexit


    # Echo time and hostname into log
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"


    # Ensure only anaconda/3 module loaded.
    module --quiet purge
    # This example uses Conda to manage package dependencies.
    # See https://docs.mila.quebec/Userguide.html#conda for more information.
    module load anaconda/3
    module load cuda/11.7

    # Creating the environment for the first time:
    # conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
   -#     pytorch-cuda=11.7 -c pytorch -c nvidia
   +#     pytorch-cuda=11.7 scipy -c pytorch -c nvidia
    # Other conda packages:
    # conda install -y -n pytorch -c conda-forge rich tqdm

    # Activate pre-existing environment.
    conda activate pytorch


   -# Stage dataset into $SLURM_TMPDIR
   -mkdir -p $SLURM_TMPDIR/data
   -cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
   -# General-purpose alternatives combining copy and unpack:
   -#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
   -#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/
   +# Prepare data for training
   +mkdir -p "$SLURM_TMPDIR/data"
   +
   +# Copy the dataset to $SLURM_TMPDIR so it is close to the GPUs for
   +# faster training
   +srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 time -p python data.py


    # Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
    unset CUDA_VISIBLE_DEVICES

    # Execute Python script
   -python main.py
   +srun python main.py


**main.py**

.. code:: python

   """Data example."""
   import logging
   import os
   import time

   import rich.logging
   import torch
   from torch import Tensor, nn
   from torch.nn import functional as F
   from torch.utils.data import DataLoader, random_split
   from torchvision import transforms
   from torchvision.datasets import INaturalist
   from torchvision.models import resnet18
   from tqdm import tqdm


   Logger = logging.getLogger(__name__)


   def main():
       training_epochs = 1
       learning_rate = 5e-4
       weight_decay = 1e-4
       batch_size = 512

       # Check that the GPU is available
       assert torch.cuda.is_available() and torch.cuda.device_count() > 0
       device = torch.device("cuda", 0)

       # Setup logging (optional, but much better than using print statements)
       logging.basicConfig(
           level=logging.INFO,
           handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
       )

       # Create a model and move it to the GPU.
       model = resnet18(num_classes=10000)
       model.to(device=device)

       optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

       # Setup ImageNet
       num_workers = get_num_workers()
       dataset_path = f"{os.environ['SLURM_TMPDIR']}/data"
       train_dataset, valid_dataset, test_dataset = make_datasets(dataset_path)
       train_dataloader = DataLoader(
           train_dataset,
           batch_size=batch_size,
           num_workers=num_workers,
           shuffle=True,
       )
       valid_dataloader = DataLoader(  # NOTE: Not used in this example.
           valid_dataset,
           batch_size=batch_size,
           num_workers=num_workers,
           shuffle=False,
       )
       test_dataloader = DataLoader(   # NOTE: Not used in this example.
           test_dataset,
           batch_size=batch_size,
           num_workers=num_workers,
           shuffle=False,
       )

       # Checkout the "checkpointing and preemption" example for more info!
       Logger.debug("Starting training from scratch.")

       # warm-up
       for i, batch in enumerate(train_dataloader):
           # Move the batch to the GPU before we pass it to the model
           batch = tuple(item.to(device) for item in batch)

           if i >= 20:
               break

       for epoch in range(training_epochs):
           Logger.debug(f"Starting epoch {epoch}/{training_epochs}")

           # Set the model in training mode (this is important for e.g. BatchNorm and Dropout layers)
           model.train()

           # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
           progress_bar = tqdm(
               total=len(train_dataloader),
               desc=f"Train epoch {epoch}",
           )

           # Training loop
           n_samples = 0
           waiting_for_data_time = 0
           end = time.time()
           for batch in train_dataloader:
               # Move the batch to the GPU before we pass it to the model
               batch = tuple(item.to(device) for item in batch)
               waiting_for_data_time += time.time() - end
               x, y = batch

               # Forward pass
               logits: Tensor = model(x)

               loss = F.cross_entropy(logits, y)

               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               # measure the elapsed time between 2 batches. This is the time we
               # wait for the data
               end = time.time()

               n_samples += y.shape[0]

               # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
               progress_bar.update(1)
               progress_bar.set_postfix({"Waiting for data (items/sec)":n_samples / waiting_for_data_time})
           progress_bar.close()

       print("Done!")


   def make_datasets(
       dataset_path: str,
       val_split: float = 0.1,
       val_split_seed: int = 42,
   ):
       """Returns the training, validation, and test splits for iNat.

       NOTE: We use the same image transforms here for train/val/test just to keep things simple.
       Having different transformations for train and validation would complicate things a bit.
       Later examples will show how to do the train/val/test split properly when using transforms.
       """
       train_dataset = INaturalist(
           root=dataset_path,
           transform=transforms.Compose([
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
           ]),
           version="2021_train"
       )
       test_dataset = INaturalist(
           root=dataset_path,
           transform=transforms.Compose([
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
           ]),
           version="2021_valid"
       )
       # Split the training dataset into a training and validation set.
       train_dataset, valid_dataset = random_split(
           train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
       )
       return train_dataset, valid_dataset, test_dataset


   def get_num_workers() -> int:
       """Gets the optimal number of DatLoader workers to use in the current job."""
       if "SLURM_CPUS_PER_TASK" in os.environ:
           return int(os.environ["SLURM_CPUS_PER_TASK"])
       if hasattr(os, "sched_getaffinity"):
           return len(os.sched_getaffinity(0))
       return torch.multiprocessing.cpu_count()


   if __name__ == "__main__":
       main()


**data.py**

.. code:: python

   """Make sure the data is available"""
   import os
   import shutil
   import sys
   import time
   from multiprocessing import Pool
   from pathlib import Path

   from torchvision.datasets import INaturalist


   def link_file(src: Path, dest: Path) -> None:
       src.symlink_to(dest)


   def link_files(src: Path, dest: Path, workers: int = 4) -> None:
       os.makedirs(dest, exist_ok=True)
       with Pool(processes=workers) as pool:
           for path, dnames, fnames in os.walk(str(src)):
               rel_path = Path(path).relative_to(src)
               fnames = map(lambda _f: rel_path / _f, fnames)
               dnames = map(lambda _d: rel_path / _d, dnames)
               for d in dnames:
                   os.makedirs(str(dest / d), exist_ok=True)
               pool.starmap(
                   link_file,
                   [(src / _f, dest / _f) for _f in fnames]
               )


   if __name__ == "__main__":
       src = "/network/datasets/inat"
       workers = os.environ["SLURM_JOB_CPUS_PER_NODE"]
       # Referencing $SLURM_TMPDIR here instead of job.sh makes sure that the
       # environment variable will only be resolved on the worker node (i.e. not
       # referencing the $SLURM_TMPDIR of the master node)
       dest = Path(os.environ["SLURM_TMPDIR"]) / "data"

       start_time = time.time()

       link_files(src, dest, workers)

       # Torchvision expects these names
       shutil.move(dest / "train.tar.gz", dest / "2021_train.tgz")
       shutil.move(dest / "val.tar.gz", dest / "2021_valid.tgz")

       INaturalist(root=dest, version="2021_train", download=True)
       INaturalist(root=dest, version="2021_valid", download=True)

       seconds_spent = time.time() - start_time

       print(f"Prepared data in {seconds_spent/60:.2f}m")


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
