# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from models.tts.fastspeech2.fs2_trainer import FastSpeech2Trainer
from models.tts.vits.vits_trainer import VITSTrainer
from models.tts.valle.valle_trainer import VALLETrainer
from models.tts.naturalspeech2.ns2_trainer import NS2Trainer
from models.tts.valle_v2.valle_ar_trainer import ValleARTrainer as valle_simple_ar
from models.tts.valle_v2.valle_nar_trainer import ValleNARTrainer as valle_simple_nar

from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        "FastSpeech2": FastSpeech2Trainer,
        "VITS": VITSTrainer,
        "VALLE": VALLETrainer,
        "NaturalSpeech2": NS2Trainer,
        "valle_simple_ar": valle_simple_ar,
        "valle_simple_nar": valle_simple_nar,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed",
        required=False,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume", action="store_true", help="The model name to restore"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Test the model"
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="resume",
        help="Resume training or finetuning.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--resume_from_ckpt_path",
        type=str,
        default="",
        help="Checkpoint for resume training or finetuning.",
    )
    # VALLETrainer.add_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print("experiment name: ", args.exp_name)
    # # CUDA settings
    cuda_relevant()

    # Build trainer
    print(f"Building {cfg.model_type} trainer")
    trainer = build_trainer(args, cfg)
    print(f"Start training {cfg.model_type} model")
    if args.test:
        trainer.test_loop()
    else:
        trainer.train_loop()


if __name__ == "__main__":
    main()
