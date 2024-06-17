import os
import torch
import json
import whisper  # pip install -U openai-whisper
import torch.nn.functional as F
import os
import librosa
import numpy as np
from tqdm import tqdm
from .speaker_verification import init_model
from evaluate import load
from modelscope.pipelines import pipeline  # run !pip install -U funasr modelscope
from modelscope.utils.constant import Tasks  # run !pip install -U funasr modelscope
import whisper  # pip install -U openai-whisper
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import torchaudio

WHISPER_PATH = "/mnt/petrelfs/hehaorui/data/pretrained_models/whisper/large-v2.pt"
WAVLM_LARGE_FINTUNED_PATH = (
    "/mnt/petrelfs/hehaorui/data/pretrained_models/wavlm/wavlm_large_finetune.pth"
)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) * 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_fid_score(target_wav, reference_wav, inference_pipeline):
    """
    Calculate the cosine similarity between emotion embeddings of two waveforms.
    """
    results = inference_pipeline([target_wav, reference_wav], granularity="frame")
    rec_target = results[0]["feats"]  # (T,768)
    rec_reference = results[1]["feats"]  # (T,768)
    # cos_sim_score = F.cosine_similarity(
    #     torch.tensor(rec_target), torch.tensor(rec_reference), dim=-1
    # )
    return calculate_fid(rec_target, rec_reference)


def extract_wavlm_similarity(target_wav, reference_wav, speaker_encoder):

    emb1 = speaker_encoder(target_wav)  # emb.shape = (batch_size, embedding_dim)
    emb1 = emb1.cpu()

    emb2 = speaker_encoder(reference_wav)  # emb.shape = (batch_size, embedding_dim)
    emb2 = emb2.cpu()

    sim = F.cosine_similarity(emb1, emb2)
    cos_sim_score = sim[0].item()
    return cos_sim_score


def calculate_wer(transcript_text, target_text, wer):
    predictions = [transcript_text]
    references = [target_text]
    wer_score = wer.compute(predictions=predictions, references=references)
    return wer_score


if __name__ == "__main__":
    from transformers import Wav2Vec2Processor, HubertForCTC

    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    import re

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    whisper_model = whisper.load_model(WHISPER_PATH)  # change to large-v2 or v3

    # Load model [Need to change]
    from models.tts.valle_x.valle_inference import ValleInference

    tts_model = ValleInference(use_vocos=True).to(device)

    reference_floder = "Wave16k16bNormalized"
    output_floder = "gen_data"

    # Load json file
    with open(
        "ref_dur_3_test_merge_1pspk_with_punc_refmeta_normwav_fix_refuid_new_diffprompt.json",
        "r",
    ) as f:
        # with open('./librispeech_ref_dur_3_test_full_with_punc_wdata.json', 'r') as f:
        json_data = f.read()
    data = json.loads(json_data)
    test_data = data["test_cases"]

    # load wer
    print("Loading WER")
    wer = load("wer")

    # load wavlm-large-fintuned
    print("Loading WavLM-large-finetuned")
    speaker_encoder = init_model(checkpoint=WAVLM_LARGE_FINTUNED_PATH).to(device).eval()

    # build emo2vec pipeline
    fid_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_base",
        model_revision="v2.0.4",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    wer_scores = []
    similarity_scores = []
    fid_scores = []

    for wav_info in tqdm(test_data):

        wav_path = wav_info["wav_path"].split("/")[-1]
        reference_path = os.path.join(reference_floder, wav_path)
        assert os.path.exists(reference_path), f"File {reference_path} not found"

        reference_wav_16k, _ = librosa.load(reference_path, sr=16000)
        # resample to 24k
        reference_wav = librosa.resample(
            reference_wav_16k, orig_sr=16000, target_sr=24000
        )
        reference_wav = torch.tensor(reference_wav, dtype=torch.float32).to(device)
        reference_wav_16k = torch.tensor(reference_wav_16k, dtype=torch.float32).to(
            device
        )

        source_text = wav_info["text"]
        target_text = wav_info["target_text"]

        output_file_name = wav_info["uid"] + ".wav"
        output_path = os.path.join(output_floder, output_file_name)

        from models.tts.valle_x.valle_inference import g2p, LANG2CODE

        transcript = source_text + " " + target_text
        transcript = "".join(
            e for e in transcript if e.isalnum() or e.isspace()
        ).lower()
        orig_transcript = transcript
        transcript = g2p(transcript, "en")[1]
        transcript = [LANG2CODE["en"]] + transcript
        transcript = torch.tensor(transcript, dtype=torch.long).to(device)

        # Run TTS based on own model [Need to change]
        output_wav = tts_model(
            {
                "speech": reference_wav.unsqueeze(0),
                "phone_ids": transcript.unsqueeze(0),
                "output_path": output_path,
            }
        )[..., 24000 * 3 :]
        torchaudio.save(output_path, output_wav.squeeze(0).cpu(), 24000)
        print(f"saved to {output_path}")
        print(f"original transcript: {target_text}")
        print("Transcribing")

        # resample to 16k
        output_wav = torchaudio.functional.resample(
            output_wav, orig_freq=24000, new_freq=16000
        )

        input_values = asr_processor(
            output_wav.squeeze(0).squeeze(0), return_tensors="pt"
        ).input_values
        logits = asr_model(input_values=input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_text = asr_processor.decode(predicted_ids[0])

        # WER: assert shape output_wav == (num_samples, )
        # transcript_text = whisper_model.transcribe(output_wav.squeeze(0).squeeze(0))['text']
        transcript_text = transcript_text.upper()
        transcript_text = re.sub(r"[^\w\s]", "", transcript_text)

        target_text = target_text.upper()
        target_text = re.sub(r"[^\w\s]", "", target_text)

        print(f"Transcript: {transcript_text}")
        wer_score = calculate_wer(transcript_text, target_text, wer)
        print(f"WER: {wer_score}")

        # SIM-O

        reference_wav = reference_wav_16k
        output_wav = output_wav.squeeze(0)
        print("Extracting WavLM similarity")
        # assert shape output_wav == (1, num_samples)
        sim_o = extract_wavlm_similarity(
            output_wav, reference_wav.unsqueeze(0), speaker_encoder
        )
        print(f"SIM-O: {sim_o}")
        # # FID:
        fid = calculate_fid_score(output_wav, reference_wav, fid_pipeline)
        print(f"FID: {fid}")

        wer_scores.append(wer_score)
        similarity_scores.append(sim_o)
        fid_scores.append(fid)

    print(f"WER: {np.mean(wer_scores)}")
    print(f"SIM-O: {np.mean(similarity_scores)}")
    print(f"FID: {np.mean(fid_scores)}")
