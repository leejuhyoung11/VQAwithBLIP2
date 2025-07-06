import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import os
from pathlib import Path
from collections import deque
from huggingface_hub import upload_file

class CustomTrainer:
    
    def __init__(self, model: nn.Module, optimizer, tokenizer, train_dataset, val_dataset=None, batch_size=8, save_dir="./checkpoints", repo_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True) if val_dataset else None
        
        self.scaler = torch.cuda.amp.GradScaler() 
        self.scheduler = None 

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")

        self.repo_id = repo_id

    def _forward_step(self, batch: dict, return_preds: bool = False):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(**inputs)
            loss = outputs.loss

        if return_preds:
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            
            labels = inputs['labels'].clone()
            labels[labels == -100] = self.tokenizer.pad_token_id
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            return loss, decoded_preds, decoded_labels

        return loss

    def train(self, num_epochs: int, resume_from_checkpoint: str = None):
        total_steps = len(self.train_dataloader) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
        else:
            start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                loss = self._forward_step(batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
            
            avg_train_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} | Average Train Loss: {avg_train_loss:.4f}")

            if self.val_dataloader:
                avg_val_loss = self.evaluate(epoch)
            
            self.save_checkpoint(epoch, avg_train_loss, avg_val_loss)

    def evaluate(self, epoch: int):
        
        self.model.eval()
        total_loss = 0
        
        last_n_samples = 1
        last_preds = deque(maxlen=last_n_samples)
        last_labels = deque(maxlen=last_n_samples)

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1} - Evaluating")
            for batch in progress_bar:
                loss, decoded_preds, decoded_labels = self._forward_step(batch, return_preds=True)
                total_loss += loss.item()
                
                last_preds.extend(decoded_preds)
                last_labels.extend(decoded_labels)

        avg_val_loss = total_loss / len(self.val_dataloader)
        print(f"\n--- Validation Results for Epoch {epoch+1} ---")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        print("\n--- Sample Predictions ---")
        for pred, label in zip(last_preds, last_labels):
            print(f"🔵 Pred:  {pred.strip()}")
            print(f"🟢 Label: {label.strip()}")
        print("---------------------------------------\n")

        return avg_val_loss
    
    def upload_checkpoint_to_hf(self, checkpoint_path, repo_id: str, remote_subdir: str = ""):
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise EnvironmentError("Hugging Face token not found in env variable 'HF_TOKEN'")

        remote_path = f"{remote_subdir}/{checkpoint_path.name}" if remote_subdir else checkpoint_path.name

        try:
            upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"Uploaded to Hugging Face: {repo_id}/{remote_path}")
        except Exception as e:
            print(f"Upload failed: {e}")


    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        save_path = self.save_dir / f"epoch_{epoch+1}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        trainable_param_names = {name for name, param in self.model.named_parameters() if param.requires_grad}
        trainable_state_dict = {k: v for k, v in self.model.state_dict().items() if k in trainable_param_names}

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": trainable_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        checkpoint_filename = f"checkpoint_epoch{epoch+1}.pt"
        checkpoint_path = save_path / checkpoint_filename

        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved to {checkpoint_path}, num of params {len(trainable_state_dict)}")

        if self.repo_id:
            self.upload_checkpoint_to_hf(
                checkpoint_path=checkpoint_path,
                repo_id=self.repo_id,
                remote_subdir=save_path.name 
            )

    def load_checkpoint(self, checkpoint_path: str):
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
    
    