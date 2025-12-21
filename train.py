"""
Training script for Animalese TTS.

Usage:
    python train.py --config configs/default.yaml
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from models import AnimaleseTTS, MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.discriminator import discriminator_loss, generator_loss, feature_matching_loss
from data import AnimaleseDataset, AnimaleseCollator, AudioProcessor
from data.dataset import create_dataloader
from utils import load_config, AnimaleseLoss
from utils.losses import MultiResolutionSTFTLoss


class Trainer:
    """Trainer for Animalese TTS."""

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=config["audio"]["sample_rate"],
            n_fft=config["audio"]["n_fft"],
            hop_length=config["audio"]["hop_length"],
            win_length=config["audio"]["win_length"],
            n_mels=config["audio"]["n_mels"],
            mel_fmin=config["audio"]["mel_fmin"],
            mel_fmax=config["audio"]["mel_fmax"],
        )

        # Initialize models
        self._init_models()

        # Initialize losses
        self.acoustic_loss = AnimaleseLoss(
            mel_weight=config["training"]["mel_loss_weight"],
            duration_weight=config["training"]["duration_loss_weight"],
            pitch_weight=config["training"]["pitch_loss_weight"],
            energy_weight=config["training"]["energy_loss_weight"],
        ).to(device)

        self.stft_loss = MultiResolutionSTFTLoss().to(device)

        # Initialize optimizers
        self._init_optimizers()

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float("inf")

    def _init_models(self):
        """Initialize all models."""
        cfg = self.config
        model_cfg = cfg["model"]

        # Main TTS model
        self.model = AnimaleseTTS(
            vocab_size=model_cfg["encoder"]["vocab_size"],
            encoder_embed_dim=model_cfg["encoder"]["embed_dim"],
            encoder_hidden_dim=model_cfg["encoder"]["hidden_dim"],
            encoder_n_layers=model_cfg["encoder"]["n_layers"],
            encoder_n_heads=model_cfg["encoder"]["n_heads"],
            variance_hidden_dim=model_cfg["variance_adaptor"]["hidden_dim"],
            variance_kernel_size=model_cfg["variance_adaptor"]["kernel_size"],
            decoder_hidden_dim=model_cfg["decoder"]["hidden_dim"],
            decoder_n_layers=model_cfg["decoder"]["n_layers"],
            decoder_n_heads=model_cfg["decoder"]["n_heads"],
            n_mels=cfg["audio"]["n_mels"],
            vocoder_upsample_rates=model_cfg["vocoder"]["upsample_rates"],
            vocoder_upsample_kernels=model_cfg["vocoder"]["upsample_kernel_sizes"],
            vocoder_resblock_kernels=model_cfg["vocoder"]["resblock_kernel_sizes"],
            vocoder_resblock_dilations=model_cfg["vocoder"]["resblock_dilation_sizes"],
            vocoder_initial_channel=model_cfg["vocoder"]["initial_channel"],
            dropout=model_cfg["encoder"]["dropout"],
        ).to(self.device)

        # Discriminators
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        # Print model info
        param_counts = self.model.get_num_params()
        print("\nModel Parameters:")
        for name, count in param_counts.items():
            print(f"  {name}: {count:,}")

    def _init_optimizers(self):
        """Initialize optimizers and schedulers."""
        cfg = self.config["training"]

        # Generator optimizer (TTS model)
        self.optimizer_g = optim.AdamW(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            betas=tuple(cfg["betas"]),
            weight_decay=cfg["weight_decay"],
        )

        # Discriminator optimizer
        self.optimizer_d = optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=cfg["learning_rate"],
            betas=tuple(cfg["betas"]),
            weight_decay=cfg["weight_decay"],
        )

        # Learning rate schedulers
        warmup_steps = cfg["warmup_steps"]

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.1, 0.5 * (1 + (warmup_steps - step) / (warmup_steps * 10)))

        self.scheduler_g = optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda)
        self.scheduler_d = optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.mpd.train()
        self.msd.train()

        epoch_losses = {
            "total": 0.0,
            "mel": 0.0,
            "duration": 0.0,
            "pitch": 0.0,
            "energy": 0.0,
            "gen": 0.0,
            "disc": 0.0,
            "fm": 0.0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            # Move to device
            chars = batch["chars"].to(self.device)
            char_lens = batch["char_lens"].to(self.device)
            mels = batch["mels"].to(self.device)
            mel_lens = batch["mel_lens"].to(self.device)
            pitches = batch["pitches"].to(self.device)
            energies = batch["energies"].to(self.device)
            durations = batch["durations"].to(self.device)

            has_audio = "audios" in batch
            if has_audio:
                real_audio = batch["audios"].to(self.device).unsqueeze(1)

            # Forward pass
            outputs = self.model(
                chars=chars,
                char_lens=char_lens,
                duration_target=durations,
                pitch_target=pitches,
                energy_target=energies,
                mel_lens=mel_lens,
            )

            # Compute acoustic losses
            acoustic_loss, loss_dict = self.acoustic_loss(
                mel_output=outputs["mel_output"],
                mel_output_before=outputs["mel_output_before"],
                mel_target=mels,
                duration_pred=outputs["duration_pred"],
                duration_target=durations,
                pitch_pred=outputs["pitch_pred"],
                pitch_target=pitches,
                energy_pred=outputs["energy_pred"],
                energy_target=energies,
                mel_lens=mel_lens,
                char_lens=char_lens,
            )

            fake_audio = outputs["audio"]

            # Train discriminators
            if has_audio and self.step > 1000:  # Start GAN training after warmup
                self.optimizer_d.zero_grad()

                # Match audio lengths
                min_len = min(real_audio.size(2), fake_audio.size(2))
                real_audio_matched = real_audio[:, :, :min_len]
                fake_audio_matched = fake_audio[:, :, :min_len].detach()

                # MPD
                mpd_real, mpd_real_fmaps = self.mpd(real_audio_matched)
                mpd_fake, mpd_fake_fmaps = self.mpd(fake_audio_matched)
                mpd_loss = discriminator_loss(mpd_real, mpd_fake)

                # MSD
                msd_real, msd_real_fmaps = self.msd(real_audio_matched)
                msd_fake, msd_fake_fmaps = self.msd(fake_audio_matched)
                msd_loss = discriminator_loss(msd_real, msd_fake)

                disc_loss = mpd_loss + msd_loss
                disc_loss.backward()
                self.optimizer_d.step()

                epoch_losses["disc"] += disc_loss.item()

                # Generator adversarial training
                self.optimizer_g.zero_grad()

                # Recompute discriminator outputs for generator update
                mpd_fake, mpd_fake_fmaps = self.mpd(fake_audio[:, :, :min_len])
                msd_fake, msd_fake_fmaps = self.msd(fake_audio[:, :, :min_len])

                # Generator loss
                gen_loss = generator_loss(mpd_fake) + generator_loss(msd_fake)

                # Feature matching loss
                fm_loss = (
                    feature_matching_loss(mpd_real_fmaps, mpd_fake_fmaps)
                    + feature_matching_loss(msd_real_fmaps, msd_fake_fmaps)
                )

                # STFT loss
                sc_loss, mag_loss = self.stft_loss(
                    fake_audio[:, :, :min_len], real_audio_matched
                )

                # Total generator loss
                total_loss = acoustic_loss + 0.1 * gen_loss + 2.0 * fm_loss + (sc_loss + mag_loss)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["grad_clip"]
                )
                self.optimizer_g.step()

                epoch_losses["gen"] += gen_loss.item()
                epoch_losses["fm"] += fm_loss.item()

            else:
                # Acoustic-only training (warmup)
                self.optimizer_g.zero_grad()
                acoustic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["grad_clip"]
                )
                self.optimizer_g.step()

            # Update schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()

            # Accumulate losses
            epoch_losses["total"] += loss_dict["total"].item()
            epoch_losses["mel"] += loss_dict["mel"].item()
            epoch_losses["duration"] += loss_dict["duration"].item()
            epoch_losses["pitch"] += loss_dict["pitch"].item()
            epoch_losses["energy"] += loss_dict["energy"].item()
            num_batches += 1
            self.step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "mel": f"{loss_dict['mel'].item():.4f}",
            })

            # Log to TensorBoard
            if self.step % self.config["training"]["log_every"] == 0:
                self._log_training(loss_dict)

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    def _log_training(self, loss_dict: Dict[str, torch.Tensor]):
        """Log training metrics to TensorBoard."""
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"train/{name}", value, self.step)

        self.writer.add_scalar(
            "train/lr", self.scheduler_g.get_last_lr()[0], self.step
        )

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        val_losses = {
            "total": 0.0,
            "mel": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                chars = batch["chars"].to(self.device)
                char_lens = batch["char_lens"].to(self.device)
                mels = batch["mels"].to(self.device)
                mel_lens = batch["mel_lens"].to(self.device)
                pitches = batch["pitches"].to(self.device)
                energies = batch["energies"].to(self.device)
                durations = batch["durations"].to(self.device)

                outputs = self.model(
                    chars=chars,
                    char_lens=char_lens,
                    duration_target=durations,
                    pitch_target=pitches,
                    energy_target=energies,
                    mel_lens=mel_lens,
                )

                _, loss_dict = self.acoustic_loss(
                    mel_output=outputs["mel_output"],
                    mel_output_before=outputs["mel_output_before"],
                    mel_target=mels,
                    duration_pred=outputs["duration_pred"],
                    duration_target=durations,
                    pitch_pred=outputs["pitch_pred"],
                    pitch_target=pitches,
                    energy_pred=outputs["energy_pred"],
                    energy_target=energies,
                    mel_lens=mel_lens,
                    char_lens=char_lens,
                )

                val_losses["total"] += loss_dict["total"].item()
                val_losses["mel"] += loss_dict["mel"].item()
                num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
            self.writer.add_scalar(f"val/{key}", val_losses[key], self.step)

        return val_losses

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state": self.model.state_dict(),
            "mpd_state": self.mpd.state_dict(),
            "msd_state": self.msd.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
            "scheduler_g_state": self.scheduler_g.state_dict(),
            "scheduler_d_state": self.scheduler_d.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.mpd.load_state_dict(checkpoint["mpd_state"])
        self.msd.load_state_dict(checkpoint["msd_state"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state"])
        self.scheduler_g.load_state_dict(checkpoint["scheduler_g_state"])
        self.scheduler_d.load_state_dict(checkpoint["scheduler_d_state"])
        print(f"Loaded checkpoint from {path} (epoch {self.epoch}, step {self.step})")

    def train(
        self,
        train_dataloader,
        val_dataloader: Optional = None,
        epochs: Optional[int] = None,
        resume_from: Optional[str] = None,
    ):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        epochs = epochs or self.config["training"]["epochs"]

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch(train_dataloader)
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.4f}")

            # Validate
            if val_dataloader and epoch % self.config["training"]["eval_every"] == 0:
                val_losses = self.validate(val_dataloader)
                print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f}")

                # Save best model
                if val_losses["total"] < self.best_loss:
                    self.best_loss = val_losses["total"]
                    self.save_checkpoint("best")

            # Regular checkpoint
            if epoch % self.config["training"]["save_every"] == 0:
                self.save_checkpoint(f"epoch_{epoch}")

        # Final checkpoint
        self.save_checkpoint("final")
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Animalese TTS")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/train", help="Path to training data"
    )
    parser.add_argument(
        "--val-dir", type=str, default=None, help="Path to validation data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda/cpu)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize trainer
    trainer = Trainer(config, device, args.checkpoint_dir)

    # Create dataloaders
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        audio_processor=trainer.audio_processor,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        shuffle=True,
    )

    val_dataloader = None
    if args.val_dir:
        val_dataloader = create_dataloader(
            data_dir=args.val_dir,
            audio_processor=trainer.audio_processor,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            shuffle=False,
        )

    # Train
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
