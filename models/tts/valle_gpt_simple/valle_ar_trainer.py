# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import torch
import time
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from models.base.new_trainer import BaseTrainer

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0, left_pad=False) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    left_pad:
        A boolean indicating whether to left pad the mask.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    mask = expaned_lengths >= lengths.unsqueeze(-1)

    if left_pad:
        mask = mask.flip(dims=[1])

    return mask


class ValleARTrainer(BaseTrainer):
    def __init__(self, args=None, cfg=None):
        super().__init__(args, cfg)
        from encodec import EncodecModel
        with self.accelerator.main_process_first():
            self.codec_encoder = EncodecModel.encodec_model_24khz()
            self.codec_encoder.set_target_bandwidth(6.0)
            self.codec_encoder.to(self.accelerator.device)
            self.codec_decoder = None    
        self.top1_accuracies = []
        self.top5_accuracies = []
        self.top10_accuracies = []
    def _accelerator_prepare(self):
        # if self.accelerator.is_main_process:
        #     breakpoint()
        # self.accelerator.wait_for_everyone()

        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
        )
    def _build_criterion(self):
        pass # loss is directly returned from model
    def _build_scheduler(self):
        from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.train.scheduler.warmup_steps,
            num_training_steps=self.cfg.train.scheduler.total_steps,
        )


    def _build_model(self):
        from .valle_ar import ValleAR
        return ValleAR(**self.cfg.model)

    def _train_step(self, batch):
        # inference codec
        '''Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
        speech: [B, T]
        speech_len: [B]
        phone_ids: [B, T]
        phone_lens: [B]
        '''
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            vq_id = self.codec_encoder.encode(batch['speech'].unsqueeze(1))
            vq_id = torch.cat([encoded[0] for encoded in vq_id], dim=-1).transpose(0,1)
            
            # recovered_audio = self.codec_decoder(vq_emb, vq=False)
            # torchaudio.save('a.wav', recovered_audio[0], 16000)
            # vq_id: [8, B, T//320]
            batch['speech'] = vq_id[0] # use first layer
        batch['speech_len'] = batch['speech_len'] // 320 # our codec downsamples 320x
        assert batch['speech_len'].max() <= batch['speech'].shape[1]

        assert batch['phone_ids'][0, -1] != 0

        phone_mask = 1 - make_pad_mask(batch['phone_lens'], max_len=batch['phone_ids'].size(1), left_pad=True).to(torch.long)
        speech_mask = 1 - make_pad_mask(batch['speech_len'], max_len=batch['speech'].size(1)).to(torch.long)

        out = self.model(
            phone_ids = batch['phone_ids'],
            phone_mask=phone_mask,
            target_ids = batch['speech'],
            target_mask=speech_mask,
        )
        loss = out.loss
        # if self.accelerator.is_main_process:
        #     print(loss)
        # if hasattr(out, 'top1_acc'):
        #     self.top1_accuracies.append(out.top1_acc)
        #     self.top5_accuracies.append(out.top5_acc)
        #     self.top10_accuracies.append(out.top10_acc)
        #     print(f'avgs: top1: {sum(self.top1_accuracies)/len(self.top1_accuracies)}, top5: {sum(self.top5_accuracies)/len(self.top5_accuracies)}, top10: {sum(self.top10_accuracies)/len(self.top10_accuracies)}')
        #     breakpoint()
        return loss
    
    ##########add your own dataloader to the trainer#############
    def _build_dataloader(self):
        from torch.utils.data import ConcatDataset, DataLoader
        if self.cfg.train.dataset.name == 'emilia':
            from .emilia_dataset import EmiliaDataset as VALLEDataset
            train_dataset = VALLEDataset()
        elif self.cfg.train.dataset.name == 'mls':
            from .mls_dataset import VALLEDataset as VALLEDataset
            train_dataset = VALLEDataset(self.cfg.trans_exp, resample_to_24k=True)
        elif self.cfg.train.dataset.name == 'libritts':
            from .libritts_dataset import VALLEDataset as VALLEDataset
            train_dataset = VALLEDataset(self.cfg.trans_exp)

        from .valle_collator import VALLECollator
        import numpy as np

        print('length of train_dataset:', len(train_dataset))

        collator = VALLECollator()

        if self.cfg.train.dataset.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Use Dynamic Batchsize......")
            from .mls_dataset import batch_by_size
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            np.random.shuffle(batch_sampler)
            print(batch_sampler[0])
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]
            from models.base.base_sampler import VariableSampler
            train_loader = DataLoader(
                train_dataset,
                collate_fn=collator,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=True, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
                persistent_workers=self.cfg.train.dataloader.persistent_workers,
                prefetch_factor=4,
            )
            print(f'process {self.accelerator.local_process_index} has {len(batches)} batches')
            self.accelerator.wait_for_everyone()

        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.local_process_index,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
                collate_fn=collator,
                sampler=sampler,
            )
            print(f'process {self.accelerator.local_process_index} has {len(train_loader)} batches')

        return train_loader, None
    def _test_step(self, batch):
        # inference codec
        '''Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
        speech: [B, T]
        speech_len: [B]
        phone_ids: [B, T]
        phone_lens: [B]
        '''
        import torchaudio
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            vq_id = self.codec_encoder.encode(batch['speech'].unsqueeze(1))
            vq_id = torch.cat([encoded[0] for encoded in vq_id], dim=-1).transpose(0,1)
            # recovered_audio = self.codec_decoder(vq_emb, vq=False)
            # torchaudio.save('a.wav', recovered_audio[0], 16000)
            # vq_id: [8, B, T//200]

            # vq_emb = self.codec_decoder.quantizer.vq2emb(vq=vq_id[:1], n_quantizers=1)
            # recovered_audio = self.codec_decoder(vq_emb, vq=False)
            # recovered_audio.shape: torch.Size([1, 1, 50200])

            batch['speech'] = vq_id[0] # use first layer


            # save gt
            recovered_audio = self.codec_encoder.decode([(vq_id[:1].transpose(0,1), None)])
            torchaudio.save('gt.wav', recovered_audio[0].cpu(), 24000)

            out_vq_ids = self.model.sample_hf(
                batch['phone_ids'][:1, ...],
                batch['speech'][:1, :225],
            )
            out_vq_ids = torch.cat([batch['speech'][:1, :225], out_vq_ids[:1, ...]], dim=1)

            # reconstruct form tokens
            recovered_audio = self.codec_encoder.decode([(out_vq_ids.unsqueeze(0), None)])
            torchaudio.save('a.wav', recovered_audio[0].cpu(), 24000)
            breakpoint()
            print()


    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        epoch_sum_loss = 0.0
        return epoch_sum_loss

    def _inference(self):
        pass
    def test_loop(self):
        self.model.eval()
        for batch in self.train_dataloader:
            self._test_step(batch)
