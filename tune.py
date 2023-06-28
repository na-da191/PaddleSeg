import os
import optuna
import re
# Import Optuna's visualization module
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
import matplotlib.pyplot as plt

def train_pplightSeg(trial):

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    end_lr = trial.suggest_uniform('end_lr', 0, 1e-4)
    #momentum = trial.suggest_uniform('momentum', 0.6, 0.98)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 0.001)
    optimizer = trial.suggest_categorical('type', ['SGD', 'Adam'])



    with open(f'pp_liteseg_stdc1_512x512_160K.yml', 'w') as f:
        f.write("""
batch_size: 16
iters: 10000
train_dataset:
  type: Dataset
  dataset_root: data/aeroscapes
  train_path: data/aeroscapes/ImageSets/train_paths.txt
  num_classes: 12
  mode: train
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
    - type: Normalize

val_dataset:
  type: Dataset
  dataset_root: data/aeroscapes
  val_path: data/aeroscapes/ImageSets/validation_paths.txt
  num_classes: 12
  mode: val
  transforms:
    - type: Normalize

optimizer:
  type: {}
  weight_decay: {}

lr_scheduler:
  type: PolynomialDecay
  learning_rate: {}
  end_lr: {}
  power:  0.9

loss:
  types:
  - type: OhemCrossEntropyLoss
    min_kept: 200000   # batch_size * 512* 512 // 16
  - type: OhemCrossEntropyLoss
    min_kept: 200000
  - type: OhemCrossEntropyLoss
    min_kept: 200000
  coef: [1, 1, 1]

model:
  type: PPLiteSeg
  backbone:
    type: STDC1
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz
  arm_out_chs: [32, 64, 128]
  seg_head_inter_chs: [32, 64, 64]

        """.format(optimizer,weight_decay,learning_rate, end_lr ))

    os.system(f"python tools/train.py \
        --config /content/PaddleSeg/pp_liteseg_stdc1_512x512_160K.yml\
        --do_eval \
        --save_interval 1000\
        --save_dir output/Optim/tune_{trial.number}")

    evaluation_result = os.popen(f"python tools/val.py \
        --config /content/PaddleSeg/pp_liteseg_stdc1_512x512_160K.yml\
        --model_path output/Optim/tune_{trial.number}/best_model/model.pdparams 2>&1").read()
    # Regular expression pattern to match the line that starts with 'all'
    pattern = r"mIoU:\s+(\d+\.\d+)"

    # Search for the pattern
    match = re.search(pattern, evaluation_result, re.MULTILINE)
    if match is not None:
         metric = float(match.group(1))
    else:
         print("No match found.") # You need to implement this function

    return metric

def main():
    pruner = optuna.pruners.MedianPruner()
    storage_url = "sqlite:////content/drive/MyDrive/PPlite_Optim.db"
    study_name = "ppliteSegOptim"

    study =  optuna.create_study(storage=storage_url,study_name=study_name,direction='maximize', pruner=pruner, sampler=optuna.samplers.TPESampler(), load_if_exists=True)
    study.optimize(train_pplightSeg, n_trials=1)


    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

if __name__ == "__main__":
    main()
