from numbers import Real
from typing import Any
import argparse
from dataclasses import dataclass

import pandas as pd
from minicons import scorer
from minicons.scorer import IncrementalLMScorer
from nltk import TweetTokenizer
from tqdm import tqdm


class FixedMaskedLMScorer(scorer.MaskedLMScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _iter_scalars(x):
        """Yield all leaf elements from nested tuples/lists/dicts."""
        if isinstance(x, dict):
            for v in x.values():
                yield from FixedMaskedLMScorer._iter_scalars(v)
        elif isinstance(x, (tuple, list)):
            for v in x:
                yield from FixedMaskedLMScorer._iter_scalars(v)
        else:
            yield x

    @staticmethod
    def _is_number_like(x):
        # torch/numpy scalar
        if hasattr(x, "item") and callable(x.item):
            try:
                x = x.item()
            except Exception:
                return False
        return isinstance(x, Real)

    @staticmethod
    def _to_float(x):
        if hasattr(x, "item") and callable(x.item):
            x = x.item()
        return float(x)

    @classmethod
    def _extract_token_surprisal(cls, item):
        """
        Robustly parse a token+surprisal from arbitrary minicons token_score outputs.

        Strategy:
          - token: first string-like found (or dict["token"] if present)
          - surprisal: first numeric leaf found anywhere in item
        """
        # token
        tok = None
        if isinstance(item, dict):
            tok = item.get("token") or item.get("tok") or item.get("text")
            # surprisal might be under key, but we still scan if absent
            if "surprisal" in item and cls._is_number_like(item["surprisal"]):
                return tok, cls._to_float(item["surprisal"])

        if tok is None:
            for leaf in cls._iter_scalars(item):
                if isinstance(leaf, str):
                    tok = leaf
                    break

        # surprisal: first numeric-like leaf
        s = None
        for leaf in cls._iter_scalars(item):
            if cls._is_number_like(leaf):
                s = cls._to_float(leaf)
                break

        if tok is None or s is None:
            raise ValueError(f"Could not parse (token, surprisal) from item: {item}")

        return tok, s

    def word_surprisal_mlm(self, sentence: str, tokenize_function: Any):
        enc = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        tok_strings = self.tokenizer.convert_ids_to_tokens(input_ids)

        tok_surps = self.token_score(sentence, surprisal=True)

        if isinstance(sentence, str) and len(tok_surps) == 1 and isinstance(tok_surps[0], list):
            tok_surps = tok_surps[0]

        filt_offsets = []
        for t, (a, b) in zip(tok_strings, offsets):
            if a == b:
                continue
            filt_offsets.append((a, b))

        if len(tok_surps) != len(filt_offsets):
            raise ValueError(
                f"Length mismatch: token_score returned {len(tok_surps)} items, "
                f"but tokenizer (without specials) has {len(filt_offsets)} tokens.\n"
                f"token_score[:3]={tok_surps[:3]}\n"
                f"tok_strings={tok_strings}\n"
                f"offsets={offsets}"
            )

        def get_surpr(item):
            if isinstance(item, (tuple, list)) and len(item) >= 2 and isinstance(item[1], (int, float)):
                return float(item[1])
            if isinstance(item, (tuple, list)) and len(item) >= 2 and hasattr(item[1], "item"):
                try:
                    return float(item[1].item())
                except Exception:
                    pass
            if isinstance(item, (tuple, list)):
                for v in item:
                    if isinstance(v, (int, float)):
                        return float(v)
                    if hasattr(v, "item"):
                        try:
                            return float(v.item())
                        except Exception:
                            pass
            return 0.0

        surps = [get_surpr(x) for x in tok_surps]

        words = tokenize_function(sentence)
        spans = []
        i = 0
        for w in words:
            j = sentence.find(w, i)
            if j == -1:
                continue
            spans.append((w, j, j + len(w)))
            i = j + len(w)

        word_scores = []
        for w, ws, we in spans:
            total = 0.0
            for s, (a, b) in zip(surps, filt_offsets):
                if not (b <= ws or a >= we):
                    total += s
            word_scores.append((w, total))

        return word_scores

    def word_score_tokenized(self, batch: Any, tokenize_function: Any, **kwargs):
        res = []
        for sentence in batch:
            res.append(self.word_surprisal_mlm(sentence, tokenize_function))
        return res


@dataclass
class Args:
    model: str
    model_type: str
    csv_in: str
    csv_out: str
    tokenizer: str
    device: str
    text_col: str
    target_col: str
    batch_size: int
    seq_aggr: str


def parse_args(argv=None) -> Args:
    p = argparse.ArgumentParser(
        description="Compute surprisal scores with minicons (MLM or Incremental LM) from a CSV."
    )

    p.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or local path (e.g., google/bert_uncased_L-2_H-128_A-2).",
    )
    p.add_argument(
        "--type",
        required=True,
        choices=["mlm", "inc"],
        dest="model_type",
        help="Model type: mlm (masked LM) or inc (incremental/causal LM).",
    )

    p.add_argument("--csv-in", required=True, dest="csv_in", help="Input CSV path.")
    p.add_argument("--csv-out", required=True, dest="csv_out", help="Output CSV path.")

    p.add_argument(
        "--tokenizer",
        default="nltk-tweet",
        choices=["nltk-tweet"],
        help=(
            "Tokenizer choice."
        ),
    )

    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (e.g., cuda:0, cpu). Default: cuda:0",
    )
    p.add_argument(
        "--text-col",
        default="Sentence",
        help="Name of the CSV column containing the input text. Default: Sentence",
    )
    p.add_argument(
        "--target-col",
        default="Target",
        help="Name of the CSV column containing the Target word. Default: Target",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for scoring. Default: 16",
    )

    p.add_argument(
        "--seq-aggr",
        default="mean",
        choices=["mean", "sum"],
        help=(
            "How to aggregate token-level scores across sequences."
        ),
    )

    a = p.parse_args(argv)
    return Args(**vars(a))


def process_scv(csv_in_path: str,
                csv_out_path: str,
                text_col: str,
                target_col: str,
                seq_aggr: str,
                tokenize_function: Any,
                scorer_model: Any):
    df = pd.read_csv(csv_in_path)

    if text_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV.")

    scores = []
    tok_scores = []
    target_scores = []
    for idx, row in tqdm(df.iterrows(), desc="Scoring", total=len(df)):
        text = str(row[text_col])
        target = str(row[target_col])

        score = scorer_model.sequence_score(
            [text],
            reduction=lambda x: -x.sum(0).item() if seq_aggr == "sum" else -x.mean(0).item(),
        )[0]

        tok_score = scorer_model.word_score_tokenized([text], tokenize_function,surprisal=True)[0]
        scores.append(score)
        tok_scores.append(tok_score)

        tscore = 0
        for item in tok_score:
            if item[0] == target:
                tscore = item[1]
                break
        target_scores.append(tscore)

    df["score"] = scores
    df["tscore"] = target_scores
    df["tok_scores"] = tok_scores

    df.to_csv(csv_out_path, index=False)
    print(f"Wrote output to {csv_out_path}")


def main(arg: Args):
    if arg.model_type == "mlm":
        scorer_model = FixedMaskedLMScorer(arg.model, device=arg.device)
    elif arg.model_type == "inc":
        scorer_model = IncrementalLMScorer(arg.model, device=arg.device)
    else:
        raise ValueError(f"Invalid model type: {arg.model_type}")

    if arg.tokenizer == "nltk-tweet":
        tokenize_function = TweetTokenizer().tokenize
    else:
        raise ValueError(f"Invalid tokenizer: {arg.tokenizer}")

    process_scv(csv_in_path=arg.csv_in,
                csv_out_path=arg.csv_out,
                text_col=arg.text_col,
                target_col=arg.target_col,
                seq_aggr=arg.seq_aggr,
                tokenize_function=tokenize_function,
                scorer_model=scorer_model, )


if __name__ == "__main__":
    """
    mlm_model = FixedMaskedLMScorer("google/bert_uncased_L-2_H-128_A-2", device="cuda:0")
    
    ilm_model = scorer.IncrementalLMScorer('sshleifer/tiny-gpt2', 'cuda:0')
    
    stimuli = ["It is a microspectrophotometry misunderstanding that the keys to the cabinet are on the table.",
               "The keys to the cabinet is on the table."]
    
    
    print(ilm_model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item()))
    print(ilm_model.token_score(stimuli, surprisal=True))
    
    # MLM scoring, inspired by Salazar et al., 2020
    print(mlm_model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item()))
    print(mlm_model.token_score(stimuli, surprisal=True))
    
    
    print(ilm_model.word_score_tokenized(
            stimuli,
            # bos_token=BOS,
            tokenize_function=TweetTokenizer().tokenize,
            surprisal=True,
            # bow_correction=True,
        ))
    
    
    
    print(mlm_model.word_surprisal_mlm(stimuli[0], TweetTokenizer().tokenize))
    """
    args = parse_args()
    main(args)