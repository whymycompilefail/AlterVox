#!/usr/bin/env python
"""Qwen3-TTS LoRA 微调脚本（定制角色声线）。

方案：固定锚点 clip_0002 做 ICL prompt（参考音频 + 转写），目标句 codec 全码本作标签，
双损失 = talker group-0 CE + code_predictor group1-15 CE（官方 forward_sub_talker_finetune）。
序列组装与 generate() 的 ICL 推理路径逐位对齐，推理时仍走参考音频克隆路径。

用法：<venv-qwentts>/Scripts/python.exe sft_train.py
      [--epochs 3] [--lr 1e-4] [--limit 0(全部)] [--no-audition]
输出：voices/sft_merged/（合并后 HF 模型）+ voices/sft_loss.csv + audition.wav
"""

import argparse
import csv
import os
import shutil
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

import config

BASE = config.TTS_BASE_MODEL
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "sft_dataset")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "sft_cache")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "sft_merged")
ANCHOR_ICL = (os.path.join(DATA, "clips", "clip_0003.wav"), "别看我像个孩子，我比任何一位大人都了解这个世界。")
ANCHOR_XVEC = os.path.join(DATA, "clips", "clip_0030.wav")  # 最长句 12s，x-vector 更稳
SEED = 2


def load_model():
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import (Qwen3TTSConfig, Qwen3TTSForConditionalGeneration,
                                      Qwen3TTSProcessor)
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    model = AutoModel.from_pretrained(BASE, dtype=torch.float32,
                                      attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(BASE, fix_mistral_regex=True)
    assert model.speech_tokenizer is not None, "speech_tokenizer 未加载"
    assert model.speaker_encoder is not None, "speaker_encoder 未加载"
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def encode_wav(model, wav_path):
    """单条编码，避免批量 encode 变长 padding 的不确定性。"""
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 24000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
    with torch.inference_mode():
        codes = model.speech_tokenizer.encode([wav.astype(np.float32)], sr=24000).audio_codes[0]
    return codes.cpu().numpy() if torch.is_tensor(codes) else np.asarray(codes)


def precompute_ref(model, processor, mode):
    """ICL: ref_code+ref_spk+ref_ids 三件套；xvec: 仅 ref_spk。均缓存到磁盘。"""
    os.makedirs(CACHE, exist_ok=True)
    if mode == "icl":
        anchor_wav, anchor_text = ANCHOR_ICL
        code_p = os.path.join(CACHE, "ref_code.npy")
        spk_p = os.path.join(CACHE, "ref_spk.npy")
        if os.path.exists(code_p) and os.path.exists(spk_p):
            code = torch.from_numpy(np.load(code_p)).long()
            spk = torch.from_numpy(np.load(spk_p)).float()
        else:
            wav = sf.read(anchor_wav, dtype="float32")[0]
            with torch.inference_mode():
                code = model.speech_tokenizer.encode([wav], sr=24000).audio_codes[0]
                spk = model.extract_speaker_embedding(audio=wav,
                                                      sr=model.speaker_encoder_sample_rate)
            code = code.cpu().long()
            spk = spk.detach().cpu().float().view(-1)
            np.save(code_p, code.numpy())
            np.save(spk_p, spk.numpy())
        ref_ids = processor(text=f"<|im_start|>assistant\n{anchor_text}<|im_end|>\n",
                            return_tensors="pt")["input_ids"]
        print(f"锚点(ICL): ref_code {tuple(code.shape)}, ref_spk {tuple(spk.shape)}, "
              f"ref_ids {tuple(ref_ids.shape)}")
        return {"code": code, "spk": spk, "ids": ref_ids}

    spk_p = os.path.join(CACHE, "spk_" + os.path.basename(ANCHOR_XVEC).replace(".wav", ".npy"))
    if os.path.exists(spk_p):
        spk = torch.from_numpy(np.load(spk_p)).float()
    else:
        wav = sf.read(ANCHOR_XVEC, dtype="float32")[0]
        with torch.inference_mode():
            spk = model.extract_speaker_embedding(audio=wav,
                                                  sr=model.speaker_encoder_sample_rate)
        spk = spk.detach().cpu().float().view(-1)
        np.save(spk_p, spk.numpy())
    print(f"锚点(xvec): {os.path.basename(ANCHOR_XVEC)}, ref_spk {tuple(spk.shape)}")
    return {"spk": spk}


def precompute_codes(model, rows):
    """全部目标句 codec 编码，缓存为 .npy。"""
    os.makedirs(CACHE, exist_ok=True)
    codes = {}
    for i, (name, _) in enumerate(rows):
        p = os.path.join(CACHE, name.replace(".wav", ".npy"))
        if os.path.exists(p):
            codes[name] = torch.from_numpy(np.load(p)).long()
        else:
            codes[name] = torch.from_numpy(
                encode_wav(model, os.path.join(DATA, "clips", name))).long()
            np.save(p, codes[name].numpy())
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"codec 编码 {i + 1}/{len(rows)}", flush=True)
    return codes


@torch.no_grad()
def _frame_embs(codec_emb, sub_embs, ids_2d):
    """[T, Q] 全码本 embedding 求和 -> [T, D]"""
    e = codec_emb(ids_2d[:, :1])
    for i in range(1, ids_2d.shape[1]):
        e = e + sub_embs[i - 1](ids_2d[:, i:i + 1])
    return e.sum(1)


def build_batch(model, input_id, tgt_code, ref):
    """按 generate() ICL 流式路径组装教师强制序列。"""
    talker = model.talker
    tc = model.config.talker_config
    top = model.config
    text_emb = talker.get_text_embeddings()
    codec_emb = talker.get_input_embeddings()
    sub_embs = talker.code_predictor.get_input_embeddings()

    # 头部: role(3) + [think, bos, lang, eos, spk, pad(+tts_bos)]
    role = talker.text_projection(text_emb(input_id[:, :3]))
    prefill = codec_emb(torch.tensor([[tc.codec_think_id, tc.codec_think_bos_id,
                                       tc.codec_language_id["chinese"],
                                       tc.codec_think_eos_id]]))
    pad_bos = codec_emb(torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]]))
    block = torch.cat([prefill, ref["spk"].view(1, 1, -1), pad_bos], dim=1)
    tts_bos, tts_eos, tts_pad = talker.text_projection(text_emb(torch.tensor(
        [[top.tts_bos_token_id, top.tts_eos_token_id, top.tts_pad_token_id]])
    )).chunk(3, dim=1)
    head = torch.cat([role,
                      torch.cat([tts_pad.expand(-1, block.shape[1] - 2, -1), tts_bos], dim=1)
                      + block[:, :-1]], dim=1)

    if "code" in ref:
        # ICL 段: (ref_text+text+eos) 文本嵌入 与 (codec_bos+ref帧) 声码嵌入按位求和
        text_e = talker.text_projection(text_emb(
            torch.cat([ref["ids"][:, 3:-2], input_id[:, 3:-5]], dim=-1)))
        text_e = torch.cat([text_e, tts_eos], dim=1)
        codec_e = torch.cat([codec_emb(torch.tensor([[tc.codec_bos_id]])),
                             _frame_embs(codec_emb, sub_embs, ref["code"]).unsqueeze(0)], dim=1)
        l_text, l_codec = text_e.shape[1], codec_e.shape[1]
        trailing_is_text = l_text > l_codec
        if trailing_is_text:
            icl = text_e[:, :l_codec] + codec_e
            trailing = text_e[:, l_codec:]
        else:
            icl = torch.cat([text_e, tts_pad.expand(-1, l_codec - l_text, -1)], dim=1) + codec_e
            trailing = tts_pad
    else:
        # x-vector-only 段（与 generate() 非 ICL 流式路径一致）:
        # prefill 只有首文本 token(+codec_bos)，其余文本 token+eos 逐帧叠加
        text_e = talker.text_projection(text_emb(input_id[:, 3:-5]))
        icl = text_e[:, :1] + codec_emb(torch.tensor([[tc.codec_bos_id]]))
        trailing = torch.cat([text_e[:, 1:], tts_eos], dim=1)
        trailing_is_text = True

    # 目标帧: 第 k 帧输入 = 全码本 embedding + trailing 补充；末尾 eos 帧只含 group-0
    T = tgt_code.shape[0]
    frames = _frame_embs(codec_emb, sub_embs, tgt_code)
    eos_in = codec_emb(torch.tensor([[tc.codec_eos_token_id]]))[0]  # [D]
    frames = torch.cat([frames, eos_in], dim=0)  # [T+1, D]
    m = trailing.shape[1]
    adds = torch.stack([trailing[:, k] if (trailing_is_text and k < m) else tts_pad[:, 0]
                        for k in range(T + 1)], dim=1)
    frames = frames.unsqueeze(0) + adds

    inputs_embeds = torch.cat([head, icl, frames], dim=1)
    flp = head.shape[1] + icl.shape[1]  # 第一个有标签的位置
    labels = torch.full((1, inputs_embeds.shape[1]), -100, dtype=torch.long)
    labels[0, flp:flp + T] = tgt_code[:, 0]
    labels[0, flp + T] = tc.codec_eos_token_id
    return inputs_embeds, labels, flp


def save_merged(model, out_dir):
    """合并 LoRA 并按 HF 标准命名保存（llama.cpp 转换可直接读取）。"""
    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for mod in model.modules():
        if isinstance(mod, LoraLinear):
            mod.merge()
            n += 1
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            continue
        state[k.replace(".base_layer", "")] = v.detach().clone().contiguous()
    save_file(state, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    # config.json 等辅助文件直接复制原版（transformers 对该自定义 config 的
    # save_pretrained(use_diff=True) 有 KeyError: 'dtype' 的 bug，且架构未变）
    # 拷贝 tokenizer / speech_tokenizer 等辅助文件
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in (".git", "speech_tokenizer")]
        for f in files:
            if f == "model.safetensors":
                continue
            src, dst = os.path.join(root, f), os.path.join(
                out_dir, os.path.relpath(os.path.join(root, f), BASE))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    shutil.copytree(os.path.join(BASE, "speech_tokenizer"),
                    os.path.join(out_dir, "speech_tokenizer"), dirs_exist_ok=True)
    print(f"已合并保存: {out_dir}  (merge 了 {n} 个 LoRA 层)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="每 epoch 句数上限，0=全部")
    ap.add_argument("--mode", choices=["icl", "xvec"], default="icl",
                    help="icl=带参考码+参考文本（HF 侧克隆）; xvec=仅说话人嵌入（llama.cpp 路径）")
    ap.add_argument("--no-audition", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    model, processor = load_model()

    rows = []
    with open(os.path.join(DATA, "metadata.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            rows.append((row["file"], row["text"]))
    print(f"数据集 {len(rows)} 句")

    ref = precompute_ref(model, processor, args.mode)
    codes = precompute_codes(model, rows)

    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import Linear as LoraLinear
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=r"talker\.(model|code_predictor\.model)\.layers\.\d+"
                                     r"\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj"
                                     r"|gate_proj|up_proj|down_proj)")
    model = inject_adapter_in_model(lcfg, model)
    n_lora = sum(1 for m in model.modules() if isinstance(m, LoraLinear))
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA 注入 {n_lora} 层, 可训练参数 {sum(p.numel() for p in trainable)/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.talker.train()

    loss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "sft_loss.csv")
    rng = np.random.default_rng(SEED)
    step, t_start = 0, time.time()
    log_f = open(loss_path, "w", encoding="utf-8", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "step", "talker_loss", "sub_loss", "total_loss", "sec"])
    tok = lambda s: processor(text=s, return_tensors="pt")["input_ids"]

    for ep in range(args.epochs):
        order = rng.permutation(len(rows))
        if args.limit:
            order = order[:args.limit]
        ep_loss = []
        for i, idx in enumerate(order):
            name, text = rows[idx]
            tgt = codes[name]
            input_id = tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
            x, y, flp = build_batch(model, input_id, tgt, ref)
            out = model.talker(inputs_embeds=x, labels=y, use_cache=False,
                               output_hidden_states=True)
            t0 = time.time()
            hs = out.hidden_states[0][-1][0, flp - 1: flp - 1 + tgt.shape[0], :]
            # 注意：官方 forward_finetune 内部的 loss 经 ForCausalLMLoss 会沿码组维错位
            # 一位（head j(emb g_j) 语义上预测 g_{j+1}），这里忽略其 loss，自行做无偏移 CE
            sub_logits, _ = model.talker.forward_sub_talker_finetune(tgt, hs)
            s_loss = torch.nn.functional.cross_entropy(
                sub_logits.float().transpose(1, 2), tgt[:, 1:])
            loss = out.loss + s_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ep_loss.append(loss.item())
            step += 1
            log_w.writerow([ep, step, f"{out.loss.item():.4f}", f"{s_loss.item():.4f}",
                            f"{loss.item():.4f}", f"{time.time() - t0:.2f}"])
            log_f.flush()
            if step % 10 == 0:
                avg = sum(ep_loss[-10:]) / min(len(ep_loss), 10)
                eta = (time.time() - t_start) / step * (len(order) * args.epochs - step)
                print(f"ep{ep} step{step}/{len(order) * args.epochs}  "
                      f"loss {avg:.4f} (talker {out.loss.item():.3f} + sub {s_loss.item():.3f})  "
                      f"eta {eta / 60:.0f}min", flush=True)
        print(f"epoch {ep} 平均 loss {sum(ep_loss) / len(ep_loss):.4f}", flush=True)

    log_f.close()
    model.talker.eval()
    save_merged(model, OUT)

    if not args.no_audition:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        wrapper = Qwen3TTSModel(model=model, processor=processor)
        if args.mode == "icl":
            anchor_wav, anchor_text = ANCHOR_ICL
            items = wrapper.create_voice_clone_prompt(ref_audio=anchor_wav, ref_text=anchor_text)
            kw = {"voice_clone_prompt": items}
        else:
            kw = {"ref_audio": ANCHOR_XVEC, "x_vector_only_mode": True}
        wavs, fs = wrapper.generate_voice_clone(
            text="世界因你而瑰丽。不论何时何地，我都会回应你的期待。",
            language="chinese", **kw)
        apath = os.path.join(OUT, f"audition_{args.mode}.wav")
        sf.write(apath, wavs[0], fs)
        print(f"试听已生成: {apath}")


if __name__ == "__main__":
    main()
