# Fingerprint

1. Run `scripts/separate_nth_images.py` to sample every nth frame from the dataset. 
   Make sure to change `root` variable to your dataset path and change *nth* variable to desired number.
2. Run `scripts/separate_train_val_test.py` to separate subjects into train and test.
3. Use `scripts/checks/check_timm_model_list.py` to find a suitable pre-trained model.
4. Create a copy of `configs/config_local.yaml`, say *configs/config2.yaml* so that you have a backup in case any edits to it creates some error.
5. Run `scripts/train.py --cfg configs/config2.yaml --out outputfolder --skip-ddp` to train the model specified in *configs/config2.yaml*

### Important values in config.yaml  
`MODEL.BACKBONE.NAME` > name of the model from the timm model list  
`DATA.TRAIN.KWARGS.batch_size` > batch size for training  
`INPUT.IMAGE.IMG_SIZE` > final image size before inputting into the model  
`OPTIM.OPTIMIZER.KWARGS.lr` > learning rate  
`OPTIM.LR_SCHEDULERS.SCHEDULERS.T_max` > number of itmerations after warmup  
`PARAMS.ITERS` > total number of iterations to train. 1 epoch = len(dataset) // batch_size  
`EVAL_EVERY` > number of iterations after which the model is evaluated to see change in accuracy