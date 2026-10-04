"""Gemma-4-E4B に LoRA で CPT / SFT をかける学習スクリプト.

LoRA continual pretraining (CPT) and supervised fine-tuning (SFT) for google/gemma-4-E4B.

モード:
    cpt  コーパス（train + qa_eval）を BOS/EOS 付きで連結し、``--seq-len`` ごとに切って
         次トークン予測で学習する（packing）。
    sft  ``artifacts/qa/sft.jsonl`` の messages を -it モデルのチャットテンプレートで整形し、
         assistant の応答部分だけに損失をかける。

CPT → SFT は ``--merge-adapter`` に CPT の adapter を渡すと、ベースへ merge してから
新しい LoRA を付けて SFT する。

コスト算出のため、学習時間・処理トークン数・スループット・ピークメモリを
``<output_dir>/run_summary.json`` に記録する（GPU 時間 × 単価でコストを出す）。

使い方（eng-cpt/ 直下、ws3-arc の GPU0 で実行）:
    CUDA_VISIBLE_DEVICES=0 python -m train.train_lora --mode cpt --output-dir artifacts/runs/cpt-full
    CUDA_VISIBLE_DEVICES=0 python -m train.train_lora --mode sft \\
        --merge-adapter artifacts/runs/cpt-full/final --output-dir artifacts/runs/cpt-full-sft
"""

import os

# PyTorch 2.13 の torch._native が Gemma4 の RoPE（bmm_outer_product）を Triton カーネルへ
# 振り替えるが、ws3-arc（RTX PRO 6000 Blackwell / driver 575）では起動時に Illegal instruction で
# 落ちる。torch を import する前に無効化しておく（環境変数で上書き指定も可）。
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "data"
QA_DIR = ROOT / "artifacts" / "qa"
BASE_MODEL = "google/gemma-4-E4B"
CHAT_TEMPLATE_MODEL = "google/gemma-4-E4B-it"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# 言語モデル以外（視覚・音声エンコーダ）に LoRA を付けないための正規表現
LORA_TARGET_REGEX = r"^(?!.*(vision|audio)).*\.(" + "|".join(LORA_TARGETS) + r")$"
IGNORE_INDEX = -100


def load_jsonl(path):
    """JSONL を読み込む.

    Args:
        path (pathlib.Path): 入力ファイル。

    Returns:
        list[dict]: 各行の dict。
    """
    with open(path) as f:
        return [json.loads(line) for line in f]


def select_cpt_docs(train_fraction, include_qa_eval, seed):
    """CPT に使う文書を選ぶ.

    処理概要: train から ``train_fraction`` の割合を書籍ごとに固定シードで抽出し、
    ``include_qa_eval`` なら qa_eval を全量加える（4 択評価の対象知識は常に学習させる）。

    Args:
        train_fraction (float): train から使う割合（0〜1）。
        include_qa_eval (bool): qa_eval を含めるか。
        seed (int): 抽出の乱数シード。

    Returns:
        list[dict]: 選んだ文書。
    """
    rng = random.Random(seed)
    by_book = {}
    for d in load_jsonl(DATA_DIR / "train.jsonl"):
        by_book.setdefault(d["book"], []).append(d)
    docs = []
    for book_docs in by_book.values():
        k = max(1, round(len(book_docs) * train_fraction))
        docs.extend(rng.sample(book_docs, k) if k < len(book_docs) else book_docs)
    if include_qa_eval:
        docs.extend(load_jsonl(DATA_DIR / "qa_eval.jsonl"))
    rng.shuffle(docs)
    return docs


def build_cpt_dataset(tokenizer, docs, seq_len):
    """文書を連結して ``seq_len`` トークンのブロックに切る（packing）.

    Args:
        tokenizer (PreTrainedTokenizerBase): トークナイザ。
        docs (list[dict]): ``text`` を持つ文書。
        seq_len (int): 1 サンプルのトークン長。

    Returns:
        Dataset: ``input_ids`` / ``labels`` を持つデータセット。
    """
    stream = []
    for d in docs:
        ids = tokenizer(d["text"], add_special_tokens=False)["input_ids"]
        stream.extend([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id])
    n_blocks = len(stream) // seq_len
    blocks = [stream[i * seq_len : (i + 1) * seq_len] for i in range(n_blocks)]
    return Dataset.from_dict({"input_ids": blocks, "labels": blocks})


def build_sft_dataset(tokenizer, rows, max_len):
    """messages をチャットテンプレートで整形し、応答部分だけを損失対象にする.

    Args:
        tokenizer (PreTrainedTokenizerBase): チャットテンプレート設定済みのトークナイザ。
        rows (list[dict]): ``messages`` を持つ SFT データ。
        max_len (int): 最大トークン長（超えたサンプルは捨てる）。

    Returns:
        Dataset: ``input_ids`` / ``labels`` を持つデータセット。
    """
    inputs, labels = [], []
    for r in rows:
        prompt = tokenizer.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(r["messages"], tokenize=False)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(f_ids) > max_len or f_ids[: len(p_ids)] != p_ids:
            continue
        inputs.append(f_ids)
        labels.append([IGNORE_INDEX] * len(p_ids) + f_ids[len(p_ids) :])
    return Dataset.from_dict({"input_ids": inputs, "labels": labels})


def pad_collator(pad_id):
    """可変長サンプルを右パディングするコレータを返す.

    Args:
        pad_id (int): パディングトークン ID。

    Returns:
        Callable[[list[dict]], dict[str, torch.Tensor]]: コレータ。
    """

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    return collate


def load_base(model_id, merge_adapter):
    """ベースモデルを bf16 で読み込み、必要なら既存 adapter を merge する.

    Args:
        model_id (str): HF Hub のモデル ID かローカルパス。
        merge_adapter (str | None): merge する LoRA adapter のパス。

    Returns:
        PreTrainedModel: 学習対象のベースモデル。
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    if merge_adapter:
        model = PeftModel.from_pretrained(model, merge_adapter).merge_and_unload()
    return model


class ThroughputCallback(TrainerCallback):
    """学習中の処理トークン数を数え、終了時に所要時間とともに保持する.

    Attributes:
        tokens (int): 損失計算に使った（パディングを除く）入力トークン数の累計。
        start (float): 学習開始時刻（``time.time()``）。
        elapsed (float): 学習の所要秒数。
    """

    def __init__(self):
        self.tokens = 0
        self.start = 0.0
        self.elapsed = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        """開始時刻を記録する.

        Returns:
            None
        """
        torch.cuda.reset_peak_memory_stats()
        self.start = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        """所要時間を確定する.

        Returns:
            None
        """
        self.elapsed = time.time() - self.start


class CountingTrainer(Trainer):
    """バッチごとの非パディングトークン数を ``ThroughputCallback`` に加算する Trainer.

    Attributes:
        meter (ThroughputCallback): トークン数の加算先。
    """

    def __init__(self, *args, meter, **kwargs):
        super().__init__(*args, **kwargs)
        self.meter = meter

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """非パディングトークン数を数えてから通常の損失計算を行う.

        Args:
            model (torch.nn.Module): 学習中のモデル。
            inputs (dict[str, torch.Tensor]): バッチ。
            return_outputs (bool): 出力も返すか。

        Returns:
            torch.Tensor | tuple: Trainer 既定の戻り値。
        """
        self.meter.tokens += int(inputs["attention_mask"].sum()) if "attention_mask" in inputs else inputs["input_ids"].numel()
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def parse_args():
    """コマンドライン引数を解釈する.

    Returns:
        argparse.Namespace: 学習設定。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["cpt", "sft"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--merge-adapter", default=None, help="SFT 前に merge する CPT adapter")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="CPT: train から使う割合")
    ap.add_argument("--no-qa-eval", action="store_true", help="CPT: qa_eval を学習に含めない")
    ap.add_argument("--sft-limit", type=int, default=0, help="SFT: 使う件数（0 = 全件）")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None, help="既定: cpt 1e-4 / sft 2e-4")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1, help="試験用に打ち切るステップ数")
    return ap.parse_args()


def main():
    """データを用意し、LoRA を付けて学習し、adapter と run_summary.json を保存する.

    Returns:
        None
    """
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "cpt":
        docs = select_cpt_docs(args.train_fraction, not args.no_qa_eval, args.seed)
        ds = build_cpt_dataset(tokenizer, docs, args.seq_len)
        data_info = {"docs": len(docs), "train_fraction": args.train_fraction, "include_qa_eval": not args.no_qa_eval,
                     "blocks": len(ds), "tokens_per_epoch": len(ds) * args.seq_len}
    else:
        tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_MODEL).chat_template
        rows = load_jsonl(QA_DIR / "sft.jsonl")
        if args.sft_limit:
            rows = random.Random(args.seed).sample(rows, min(args.sft_limit, len(rows)))
        ds = build_sft_dataset(tokenizer, rows, args.seq_len)
        data_info = {"rows": len(rows), "samples": len(ds),
                     "tokens_per_epoch": sum(len(x) for x in ds["input_ids"]),
                     "target_tokens_per_epoch": sum(sum(t != IGNORE_INDEX for t in x) for x in ds["labels"])}
    print(json.dumps(data_info, ensure_ascii=False), flush=True)

    model = load_base(args.base_model, args.merge_adapter)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGET_REGEX, task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()

    lr = args.lr or (1e-4 if args.mode == "cpt" else 2e-4)
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="epoch", save_only_model=True, report_to=[],
        seed=args.seed, dataloader_num_workers=2, remove_unused_columns=False,
    )
    meter = ThroughputCallback()
    trainer = CountingTrainer(model=model, args=targs, train_dataset=ds, data_collator=pad_collator(tokenizer.pad_token_id),
                              callbacks=[meter], meter=meter)
    result = trainer.train()
    model.save_pretrained(out / "final")

    summary = {
        "mode": args.mode, "args": vars(args), "lr": lr, "data": data_info,
        "trainable_params": trainable, "total_params": total,
        "steps": result.global_step, "train_loss": result.training_loss,
        "wall_seconds": meter.elapsed, "gpu_hours": meter.elapsed / 3600,
        "tokens_processed": meter.tokens, "tokens_per_second": meter.tokens / max(meter.elapsed, 1e-9),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_name": torch.cuda.get_device_name(0),
        "loss_history": [h for h in trainer.state.log_history if "loss" in h],
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "loss_history"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
