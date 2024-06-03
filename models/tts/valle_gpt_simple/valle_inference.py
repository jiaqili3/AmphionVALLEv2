import torch
import torchaudio

class ValleInference(torch.nn.Module):
    def __init__(self, use_vocos=False, ar_path=None, nar_path=None, device='cuda'):
        super().__init__()

        self.device = device

        from .valle_ar import ValleAR
        self.ar_model = ValleAR(
            phone_vocab_size=100,
            target_vocab_size=1024,
            pad_token_id=1124,
            bos_target_id=1125,
            eos_target_id=1126,
            bos_phone_id=1127,
            eos_phone_id=1128,
            bos_prompt_id=1129,
            eos_prompt_id=1130,
        )
        # change the following path to your trained model path
        if ar_path is not None:
            self.ar_model.load_state_dict(torch.load(ar_path, map_location='cpu'))
        else:
            try:
                from huggingface_hub import hf_hub_download
                self.ar_model.load_state_dict(torch.load(hf_hub_download('jiaqili3/vallex', 'valle_ar_mls_encodec.bin'), endpoint='https://hf-mirror.com', map_location='cpu'))
                # or load from your local path
                # self.ar_model.load_state_dict(torch.load('/mnt/petrelfs/hehaorui/jiaqi/vc-dev/ckpt/valle_gpt_simple/ar_mls/checkpoint/epoch-0005_step-0406000_loss-2.203645/valle_ar_mls_encodec.bin'))
            except Exception as e:
                raise NotImplementedError(f'No AR pretrianed model found! Original failure: {e}')
        self.ar_model.eval().to(self.device)
        from .valle_nar import ValleNAR
        self.nar_model = ValleNAR(
            phone_vocab_size=100,
            target_vocab_size=1024,
            pad_token_id=1124,
            bos_target_id=1125,
            eos_target_id=1126,
            bos_phone_id=1127,
            eos_phone_id=1128,
            bos_prompt_id=1129,
            eos_prompt_id=1130,
        )
        if nar_path is not None:
            self.nar_model.load_state_dict(torch.load(nar_path, map_location='cpu'))
        else:
            try:
                from huggingface_hub import hf_hub_download
                self.nar_model.load_state_dict(torch.load(hf_hub_download('jiaqili3/vallex', 'valle_nar_mls_encodec.bin', endpoint='https://hf-mirror.com'), map_location='cpu'))
            except Exception as e:
                raise NotImplementedError(f'No NAR pretrianed model found! Original failure: {e}')

        self.nar_model.eval().to(self.device)

        from encodec import EncodecModel
        self.codec_encoder = EncodecModel.encodec_model_24khz()
        self.codec_encoder.set_target_bandwidth(6.0)
        self.codec_encoder.to(self.device)
        if use_vocos:
            from vocos import Vocos
            self.codec_decoder = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
            self.codec_decoder.to(self.device)
            

        self.use_vocos = use_vocos
            
    def decode(self, vq_ids):
        '''vq_ids.shape: [8, B, T],
        returns: [B, 1, T*320]'''
        if not self.use_vocos:
            return self.codec_encoder.decode([(vq_ids.transpose(0,1), None)])
        else:
            features = self.codec_decoder.codes_to_features(vq_ids.squeeze(1))
            bandwidth_id = torch.tensor([2], device=vq_ids.device)
            return self.codec_decoder.decode(features, bandwidth_id=bandwidth_id).unsqueeze(0)


    def forward(self, batch, chunk_configs:list, return_prompt=False):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        with torch.no_grad():
            # inference codec
            '''Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
            speech: [B, T]
            speech_len: [B]
            phone_ids: [B, T]
            phone_lens: [B]
            '''
            vq_id = self.codec_encoder.encode(batch['speech'].unsqueeze(1))
            vq_id = torch.cat([encoded[0] for encoded in vq_id], dim=-1).transpose(0,1)
            
            chunks = chunk_configs
            
            for chunk in chunks:
                ar_vq_ids = self.ar_model.sample_hf(
                    batch['phone_ids'],
                    vq_id[0, :, :],
                    top_p=chunk['top_p'],
                    top_k=chunk['top_k'],
                    temperature=chunk['temperature'],
                    num_beams=chunk['num_beams'],
                    repeat_penalty=chunk['repeat_penalty'],
                    max_length=chunk['max_length'],
                )
                recovered_audio_ar = self.decode(ar_vq_ids.unsqueeze(0))
                torchaudio.save('recovered_audio_ar.wav', recovered_audio_ar[0].cpu(), 24000)

                nar_vq_ids = self.nar_model.sample_hf(
                    phone_ids=batch['phone_ids'],
                    prompt_ids=vq_id[:,:,:],
                    first_stage_ids=ar_vq_ids,
                    top_p=1.0,
                    top_k=40,
                    temperature=1.0,
                )

                if return_prompt:
                    nar_vq_ids = torch.cat([vq_id[..., :], nar_vq_ids], dim=-1)

                recovered_audio = self.decode(nar_vq_ids)
                return recovered_audio
