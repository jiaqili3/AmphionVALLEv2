# python -m models.tts.valle_gpt.valle_ar
from .modeling_llama import LlamaConfig, LlamaForCausalLM, LlamaModel
from .modeling_llama import CrossEntropyLoss
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
from transformers.cache_utils import Cache
from einops import rearrange
import tqdm


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ValleAR(nn.Module):
    def __init__(
        self,
        phone_vocab_size=256,
        target_vocab_size=1024,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        pad_token_id=1281,
        bos_target_id=1282,
        eos_target_id=1025,
        bos_phone_id=1284,
        eos_phone_id=1285,
        use_input_embeds=False,
        emb_dim=256,
        num_prediction_heads=1,
        **kwargs,
    ):
        super(ValleAR, self).__init__()
        self.phone_vocab_size = phone_vocab_size
        target_vocab_size = target_vocab_size + 10
        self.config = LlamaConfig(
            vocab_size=phone_vocab_size + target_vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            pad_token_id=pad_token_id,
            bos_token_id=bos_target_id,
            eos_token_id=eos_target_id,
        )
        self.target_vocab_size = target_vocab_size
        self.pad_token_id = pad_token_id
        self.bos_target_id = bos_target_id
        self.eos_target_id = eos_target_id
        self.bos_phone_id = bos_phone_id
        self.eos_phone_id = eos_phone_id
        self.model = LlamaModel(self.config)
        # disable llama internal embeddings
        self.model.embed_tokens = None

        self.num_prediction_heads = num_prediction_heads
        print(f"num_prediction_heads={num_prediction_heads}")

        self.emb_text = nn.Embedding(phone_vocab_size + 10, hidden_size)
        self.emb_code = nn.ModuleList(
            [
                nn.Embedding(target_vocab_size, hidden_size)
                for i in range(num_prediction_heads)
            ]
        )

        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, 768),
                    nn.GELU(),
                    nn.Linear(768, target_vocab_size),
                )
                for _ in range(num_prediction_heads)
            ]
        )

        self.emb_text.apply(initialize_weights)
        self.emb_code.apply(initialize_weights)
        self.predict_layer.apply(initialize_weights)

    def forward(self, phone_ids, phone_mask, target_ids, target_mask):
        """target_ids: [Q, B, T]"""
        phone_ids, phone_mask, phone_label = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        phone_ids = (
            phone_ids - self.target_vocab_size
        )  # since we added this in the inner function

        # embed the phone
        emb_text = self.emb_text(phone_ids)

        # pad eos token in the code
        target_ids, target_mask, target_labels = self.add_target_eos_label(
            target_ids,
            target_mask,
            self.eos_target_id,
            pad_token_id=0,
        )

        # embed the target
        emb_code = [
            self.emb_code[i](target_ids[i]) for i in range(self.num_prediction_heads)
        ]
        emb_code = torch.stack(emb_code, 2).sum(2)

        attention_mask = torch.cat([phone_mask, target_mask], 1)

        inputs_embeds = torch.cat([emb_text, emb_code], 1)  # [B, T, H]

        model_input = {}
        model_input["inputs_embeds"] = inputs_embeds
        model_input["input_ids"] = None  # only send in embedding
        model_input["attention_mask"] = attention_mask

        # process `target_ids`
        # target_ids = target_ids * target_mask + (-100) * (1-target_mask)

        # input_token_ids = torch.cat([phone_ids, target_ids], dim=-1)
        # attention_mask = torch.cat([phone_mask, target_mask], dim=-1)
        # breakpoint()

        out = self.model(
            **model_input,
            return_dict=True,
        )

        hidden_states = out[0]
        logits = [pred(hidden_states).float() for pred in self.predict_layer]

        total_losses = []
        target_len = target_ids.shape[-1]  # length of audio codes

        for i in range(self.num_prediction_heads):
            labels = target_labels[i]
            # Shift so that tokens < n predict n
            shift_logits = logits[i][..., -target_len:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.target_vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            total_losses.append(loss_fct(shift_logits, shift_labels))

        loss_weights = [1.0, 0.5]
        total_loss = sum(
            [
                loss_weights[i] * total_losses[i]
                for i in range(self.num_prediction_heads)
            ]
        )
        out.loss = total_loss

        # calcualte top1, top5, top10 accuracy
        # logits = out.logits
        # logits = logits[:, -target_ids.shape[1] :]
        # top1_acc = (logits.argmax(-1)[...,:-1] == target_ids[:, 1:])
        # top1_acc = (top1_acc * target_mask[...,:-1]).sum() / target_mask.sum()

        # top5_acc = torch.topk(logits[..., :-1, :], 5, dim=-1)[1]
        # top5_acc = (top5_acc == target_ids[:, 1:].unsqueeze(-1))
        # top5_acc = (top5_acc * target_mask[...,:-1].unsqueeze(-1)).sum() / target_mask.sum()

        # top10_acc = torch.topk(logits[..., :-1, :], 10, dim=-1)[1]
        # top10_acc = (top10_acc == target_ids[:, 1:].unsqueeze(-1))
        # top10_acc = (top10_acc * target_mask[...,:-1].unsqueeze(-1)).sum() / target_mask.sum()

        # out.top1_acc = top1_acc
        # out.top5_acc = top5_acc
        # out.top10_acc = top10_acc

        return out

    def add_phone_eos_bos_label(
        self, phone_ids, phone_mask, phone_eos_id, phone_bos_id, pad_token_id
    ):
        # phone_ids: [B, T]
        # phone_mask: [B, T]

        phone_ids = phone_ids + self.target_vocab_size * phone_mask

        phone_ids = phone_ids * phone_mask
        phone_ids = F.pad(phone_ids, (0, 1), value=0) + phone_eos_id * F.pad(
            1 - phone_mask, (0, 1), value=1
        )  # make pad token eos token, add eos token at the end
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add eos mask
        phone_ids = phone_ids * phone_mask + pad_token_id * (
            1 - phone_mask
        )  # restore pad token ids
        phone_ids = F.pad(phone_ids, (1, 0), value=phone_bos_id)  # add bos token
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add bos mask
        phone_label = -100 * torch.ones_like(
            phone_ids
        )  # loss for entire phone is not computed (passed to llama)
        return phone_ids, phone_mask, phone_label

    def add_target_eos_label(
        self, target_ids, target_mask, target_eos_id, pad_token_id
    ):
        """add eos token to the audio codes"""
        # target_ids: [Q, B, T]
        # target_mask: [B, T]
        target_ids = target_ids * target_mask.expand(target_ids.shape[0], -1, -1)
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        ).expand(target_ids.shape[0], -1, -1)
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_label = target_ids * target_mask + (-100) * (
            1 - target_mask
        )  # loss for target is computed on unmasked tokens
        return target_ids, target_mask, target_label

    def add_target_eos_bos_label(
        self, target_ids, target_mask, target_eos_id, target_bos_id, pad_token_id
    ):
        # target_ids: [B, T]
        # target_mask: [B, T]
        target_ids = target_ids * target_mask
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        )
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_ids = F.pad(target_ids, (1, 0), value=target_bos_id)
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_label = target_ids * target_mask + (-100) * (
            1 - target_mask
        )  # loss for target is computed on unmasked tokens
        return target_ids, target_mask, target_label

    def get_emb(self, phone_ids, target_ids):
        phone_embs = self.emb_text(phone_ids)
        emb_code = [
            self.emb_code[i](target_ids[i]) for i in range(self.num_prediction_heads)
        ]
        emb_code = torch.stack(emb_code, 2).sum(2)  # [B, T, H]
        return torch.cat([phone_embs, emb_code], 1)

    @torch.no_grad()
    def generate(
        self,
        # emb,
        phone_ids,  # [B,T]
        prompt_ids,  # [Q,B,T]
        phone_mask=None,
        max_length=2000,
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        repeat_penalty=1.0,
        attention_mask=None,
        max_new_token=2048,
        min_new_token=0,
        LogitsWarpers=[],
        LogitsProcessors=[],
        infer_text=False,
        return_attn=False,
        return_hidden=False,
        return_dict=False,
        include_prompt=False,
    ):
        from utils.topk_sampling import top_k_top_p_filtering

        prompt_ids = prompt_ids[: self.num_prediction_heads]
        eos_token = self.eos_target_id
        model_input = {}
        if phone_mask is None:
            phone_mask = torch.ones_like(phone_ids, dtype=torch.long)
        phone_ids, phone_mask, _ = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        phone_ids = (
            phone_ids - self.target_vocab_size
        )  # since we added this in the inner function

        emb = self.get_emb(phone_ids, prompt_ids)

        attentions = []
        hiddens = []

        finish = torch.zeros(phone_ids.shape[0], device=phone_ids.device).bool()

        # temperature = temperature[None].expand(phone_ids.shape[0], -1)
        # temperature = rearrange(temperature, "b n -> (b n) 1")

        # attention_mask_cache = torch.ones((inputs_ids.shape[0], inputs_ids.shape[1]+max_new_token,), dtype=torch.bool, device=inputs_ids.device)
        # if attention_mask is not None:
        #     attention_mask_cache[:, :attention_mask.shape[1]] = attention_mask

        start_idx = prompt_ids.shape[-1]
        end_idx = torch.zeros(
            prompt_ids.shape[-1], device=prompt_ids.device, dtype=torch.long
        )

        for i in tqdm.tqdm(range(max_new_token)):
            # model_input = self.prepare_inputs_for_generation(inputs_ids,
            #     outputs.past_key_values if i!=0 else None,
            #     attention_mask_cache[:, :inputs_ids.shape[1]], use_cache=True)
            model_input["past_key_values"] = outputs.past_key_values if i != 0 else None
            model_input["use_cache"] = True

            if i == 0:
                model_input["inputs_embeds"] = emb
            else:
                emb = self.get_emb(phone_ids, prompt_ids)
                model_input["inputs_embeds"] = emb

            model_input["input_ids"] = None
            outputs = self.model.forward(**model_input, output_attentions=return_attn)
            attentions.append(outputs.attentions)
            hidden_states = outputs[0]
            if return_hidden:
                hiddens.append(hidden_states[:, -1])

            logits = torch.stack(
                [
                    self.predict_layer[i](hidden_states)
                    for i in range(self.num_prediction_heads)
                ],
                3,
            )
            # logits: [B, T, H, Q]
            logits = logits[:, -1].float()

            logits = rearrange(logits, "b c n -> (b n) c")

            logits = logits / temperature

            # top p and top k sampling
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            logits_token = prompt_ids[0, :, start_idx:]  # for repetition penalty calc

            for logitsProcessors in LogitsProcessors:
                logits = logitsProcessors(logits_token, logits)

            for logitsWarpers in LogitsWarpers:
                logits = logitsWarpers(logits_token, logits)

            if i < min_new_token:
                logits[:, eos_token] = -torch.inf

            scores = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(scores, num_samples=1)

            idx_next = rearrange(
                idx_next, "(b q) 1 -> b q 1", q=self.num_prediction_heads
            )
            idx_next = rearrange(idx_next, "b q 1 -> q b 1")
            finish = finish | (idx_next == eos_token).any(1)
            if finish.any():
                # if any vq layer prediction finishes, break out
                break
            prompt_ids = torch.cat([prompt_ids, idx_next], -1)

            end_idx = end_idx + (~finish).int()

        if return_hidden:
            hiddens = torch.stack(hiddens, 1)
            hiddens = [hiddens[idx, :i] for idx, i in enumerate(end_idx.int())]

        if not finish.all():
            print(f"Incomplete result. Not all are finished")
            # print(f'Incomplete result. hit max_new_token: {max_new_token}')

        if not include_prompt:
            prompt_ids = prompt_ids[..., start_idx:]  # [Q,B,T]

        if not return_dict:
            return prompt_ids
        else:
            return {
                "ids": prompt_ids,
                "attentions": attentions,
                "hiddens": hiddens,
            }

    # def prepare_inputs_for_generation(
    #     self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    # ):
    #     '''input_ids: [B,T,Q]'''
    #     # With static cache, the `past_key_values` is None
    #     # TODO joao: standardize interface for the different Cache classes and remove of this if
    #     has_static_cache = False
    #     if past_key_values is None:
    #         past_key_values = getattr(self.gpt.layers[0].self_attn, "past_key_value", None)
    #         has_static_cache = past_key_values is not None

    #     past_length = 0
    #     if past_key_values is not None:
    #         if isinstance(past_key_values, Cache):
    #             past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
    #             max_cache_length = (
    #                 torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
    #                 if past_key_values.get_max_length() is not None
    #                 else None
    #             )
    #             cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
    #         # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
    #         else:
    #             cache_length = past_length = past_key_values[0][0].shape[2]
    #             max_cache_length = None

    #         # Keep only the unprocessed tokens:
    #         # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
    #         # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
    #         # input)
    #         if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
    #             input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
    #         # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
    #         # input_ids based on the past_length.
    #         elif past_length < input_ids.shape[1]:
    #             input_ids = input_ids[:, past_length:]
    #         # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

    #         # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
    #         if (
    #             max_cache_length is not None
    #             and attention_mask is not None
    #             and cache_length + input_ids.shape[1] > max_cache_length
    #         ):
    #             attention_mask = attention_mask[:, -max_cache_length:]

    #     position_ids = kwargs.get("position_ids", None)
    #     if attention_mask is not None and position_ids is None:
    #         # create position_ids on the fly for batch generation
    #         position_ids = attention_mask.long().cumsum(-1) - 1
    #         position_ids.masked_fill_(attention_mask == 0, 1)
    #         if past_key_values:
    #             position_ids = position_ids[:, -input_ids.shape[1] :]

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
    #         # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
    #         # TODO: use `next_tokens` directly instead.
    #         model_inputs = {"input_ids": input_ids.contiguous()}

    #     input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    #     if cache_position is None:
    #         cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
    #     else:
    #         cache_position = cache_position[-input_length:]

    #     if has_static_cache:
    #         past_key_values = None

    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "cache_position": cache_position,
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #         }
    #     )
    #     return model_inputs

    def sample_hf(
        self,
        phone_ids,  # the phones of prompt and target should be concatenated together
        prompt_ids,
        inputs_embeds=None,
        max_length=2000,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        repeat_penalty=1.0,
        num_beams=1,
    ):
        if inputs_embeds is not None:
            inputs_embeds = self.emb_linear(inputs_embeds)
        phone_mask = torch.ones_like(phone_ids)
        prompt_mask = torch.ones_like(prompt_ids)
        phone_ids, _, _ = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        prompt_ids, _, _ = self.add_target_eos_bos_label(
            prompt_ids,
            prompt_mask,
            self.eos_target_id,
            self.bos_target_id,
            self.pad_token_id,
        )
        prompt_ids = prompt_ids[:, :-1]  # remove end token. Make it continue mode

        input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1)

        if inputs_embeds is not None:
            raise NotImplementedError
            inputs_embeds = torch.cat(
                [inputs_embeds, self.model.model.embed_tokens(input_token_ids)], dim=1
            )
            generated_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                do_sample=True,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_target_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
            )
            gen_tokens = generated_ids[:, :-1]
            return gen_tokens

        input_length = input_token_ids.shape[1]
        generated_ids = self.model.generate(
            input_token_ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_target_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            num_beams=num_beams,
        )

        gen_tokens = generated_ids[:, input_length:-1]

        return gen_tokens


def test():
    model = ValleAR(num_prediction_heads=2)

    phone_ids = torch.LongTensor([[1, 2, 3, 4, 5, 0]])
    phone_mask = torch.LongTensor([[1, 1, 1, 0, 0, 0]])
    target_ids = torch.LongTensor(
        [[765, 234, 123, 234, 123, 599], [765, 234, 894, 234, 238, 599]]
    ).unsqueeze(1)
    target_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1]])

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for i in range(15):
        optimizer.zero_grad()
        out = model(
            phone_ids=phone_ids,
            phone_mask=phone_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )
        loss = out.loss

        loss.backward()

        optimizer.step()

        print(f"iter={i}, {loss}.")

    phone_ids = torch.LongTensor([1, 2, 3]).reshape(1, -1)
    target_ids = torch.LongTensor([765, 234]).expand(2, 1, -1)
    sampled = model.generate(phone_ids, target_ids)

    print(sampled)

    breakpoint()


if __name__ == "__main__":
    test()
