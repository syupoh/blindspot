# Blindspot Monitoring in esophagogastroduodenoscopy(EGD)

## Dependency
```
- python 3.7.8
- CUDA 10.1
```

```bash
>> pip install -r requirements.txt
```

<!-- ## Upload Data to Server
```bash
>> pip install paramiko
```
```bash
usage: python upload.py [-h] [--source SOURCE] [--target TARGET]

optional arguments:
  -h, --help            show this help message and exit
  --source, -s SOURCE
                        base directory path
  --target, -t TARGET
                        directory path to save
``` -->

<!-- ## K-Fold Cross Validation
```bash
usage: python train_cv.py [-h] [--gpu GPU] [--train_path TRAIN_PATH]
             [--valid_path VALID_PATH] [--normalize NORMALIZE] [--scale SCALE]
             [--label_smoothing LABEL_SMOOTHING] [--ckpt_name CKPT_NAME]
             [--augmentation] [--cross_validation CROSS_VALIDATION]
             [--batch BATCH] [--evaluate EVALUATE]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU, -g GPU     select gpu to train
  --train_path TRAIN_PATH, -t TRAIN_PATH
                        base path of data to train
  --valid_path VALID_PATH, -v VALID_PATH
                        base path of data to valid
  --normalize NORMALIZE
                        normalize in preprocess
  --scale SCALE         scale in preprocess
  --label_smoothing LABEL_SMOOTHING, -ls LABEL_SMOOTHING
                        value of label smoothing
  --ckpt_name CKPT_NAME
                        checkpoint name to save model
  --augmentation        use augmentation data
  --cross_validation CROSS_VALIDATION, -cv CROSS_VALIDATION
                        Do Cross Validation
  --batch BATCH         set batch size
  --evaluate EVALUATE   do evaluate
```
 -->


## Train Model
```bash
usage: python train.py [-h] [--gpu int] [--train_path TRAIN_PATH]
                [--scale] [--label_smoothing float]
                [--ckpt_name CKPT_NAME] 
                [--cross_validation] [--batch int]
                [--evaluate ]

optional arguments:
  -h, --help            show this help message and exit
  --gpu, -g int     select gpu number to train
  --train_path, -t TRAIN_PATH
                        base path of train data
  --scale                scaling in preprocess
  --label_smoothing, -ls float, default=0.1
                        value of label smoothing
  --ckpt_name CKPT_NAME
                        checkpoint name to save model
  --cross_validation, -cv, default=True
                        Do Cross Validation
  --batch int, default=4         set batch size
  --evaluate, default=True  do evaluation
```


## Evaluate
```bash
usage: python evaluate.py [-h] [--gpu int] [--val_data VAL_DATA]
                   [--scale]
                   [--label_smoothing float] [--ckpt_path CKPT_PATH]
                   [--batch int] [--cam]

optional arguments:
  -h, --help            show this help message and exit
  --gpu, -g int     select gpu number to train
  --val_data VAL_DATA
                        base path of data to evaluate
  --scale               scale in preprocess
  --label_smoothing, -ls float, default=0.1
                        value of label smoothing
  --ckpt_path CKPT_PATH
                        checkpoint path to load
  --batch int, default=4         set batch size
  --cam                 generate Grad CAM image
```