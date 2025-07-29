# **🚀 How to Run This Project (Reproducibility Guide)**

Follow the steps below to fully reproduce the results in this repository.

## 1. Create Conda Environment (Python 3.11)

```markdown
conda create -n blip2env python=3.11
conda activate blip2env
```

## 2. Run Setup Script

```markdown
bash setup.sh
```

## 3. Prepare Test Dataset

```python
./datasets/test.csv
./datasets/sample_submission.csv
./datasets/test_input_images/
```

## 4. How to Train

| Stage | Script Name       | Description                                              |
| ----- | ----------------- | -------------------------------------------------------- |
| 1     | train_script_1.py | Image Captioning training with Various format            |
| 2     | train_script_2.py | Initial Open-Ended VQA Training                          |
| 3     | train_script_3.py | Open-Ended VQA Training for Reasoning Enhancement        |
| 4     | train_script_4.py | Closed-Ended Training for Multiple Choice Adaptation     |

```python
# Train stage 1
python3 ./src/train_script_stage1.py
# Train stage 2
python3 ./src/train_script_stage2.py
# Train stage 3
python3 ./src/train_script_stage3.py
# Train stage 4
python3 ./src/train_script_stage4.py

```

### **🛑**  Early Stopping Policy

While each training script is configured to run for a predefined number of epochs (e.g., 50), we observed signs of overfitting starting at specific epochs during each stage. Therefore, we manually applied early stopping based on validation performance.

The decision to stop training was made by:
• **Monitoring the validation loss and/or accuracy across epochs**
• **Identifying the epoch where the validation performance peaked before starting to degrade**

Based on this manual inspection, we selected the following checkpoints for each stage:

```python
| Stage | Selected Epoch |
|-------|----------------|
| 1     | Epoch 12       |
| 2     | Epoch 3        |
| 3     | Epoch 3        |
| 4     | Epoch 4        |
```

### Resuming Training from a Checkpoint

If training is interrupted or you want to continue from a previously saved state, the training scripts are configured to resume from a given checkpoint path. To resume training, specify the path to the checkpoint inside the script file when calling the `train()` method:

```python
# Resuming from epoch5
trainer.train(
    num_epochs=config['training']['stage1']['num_epochs'],
    resume_from_checkpoint='../outputs/checkpoints/ImageCaptioning-stage1/
    epoch_5/checkpoint_epoch5.pt'
)
```

### Checkpoint Output Locations

After each training stage is completed, the corresponding model checkpoint is saved automatically to the path specified in the config file. The expected output locations for each stage are defined in the config as follows:

```python
# Example (./configs/config.yaml)
path:
	stage1_checkpoint: "../outputs/checkpoints/ImageCaptioning-stage1/epoch_12/checkpoint_epoch12.pt"
	stage2_checkpoint: "../outputs/checkpoints/ImageCaptioning-stage2/epoch_3/checkpoint_epoch3.pt"
	stage3_checkpoint: "../outputs/checkpoints/ImageCaptioning-stage3/epoch_3/checkpoint_epoch3.pt"
	stage4_checkpoint: "../outputs/checkpoints/ImageCaptioning-stage4/epoch_4/checkpoint_epoch4.pt"
```

## 5. Inference

The inference process is handled by the _inference.py_ script.

You can run inference using:

```python
python3 ./src/inference.py
```

The generated prediction file will be saved to:

```python
'../outputs/predicitons/'
```
