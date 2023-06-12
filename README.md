# **Pressure-Ulcer-Staging-pytorch**

![Untitled](https://github.com/yunvleetp/Pressure-Ulcer-Staging-pytorch/assets/49764894/c4489a25-9525-415f-823b-03b2409359bd)

- A convolutional neural network (CNN) model was developed to classify **pressure injuries and related dermatoses** using 3,098 clinical images.
- The study investigated whether implementing the CNN could improve initial pressure injury classification decisions made by physicians.
- In order to evaluate the extent to which AI assistance improves the accuracy of medical diagnoses, we conducted a survey among dermatology residents, ward nurses, and medical students.

# Requirement

The repository requirements are documented in req.txt. The experimental setup utilized a Tesla V100 GPU with 32GB of VRAM and Ubuntu 18.04.6 LTS as the operating system.

# Usage

```bash
## Training code
python3 /send/fdgClass/pp/code/classify.py --gpu_ids 0,1,2,3 --dataroot dir_to_data --checkpoints_dir dir_to_model_checkpoint --input_nc 3 --output_nc num_of_class --name exp_name --fold kfold_fold_num --n_epochs 10 --n_epochs_decay 190 --lr 0.00001 --batchSize 48 --depthSize 512 --model SEResNext101 --aug --class7

## Evaluation code
python3 confMatrix.py
python3 gradcam.py
```

# Reference

- Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 7132-7141. 2018.
