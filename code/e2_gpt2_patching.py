from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from cif import BettingCSTracker, cs_radius
from utils import ensure_dir, figures_dir, get_device, results_dir, set_seed


def _placeholder_outputs(reason: str) -> None:
    print(f"[E2] Skipping GPT-2 experiment: {reason}")
    ensure_dir(results_dir())

    out_cs = results_dir() / "e2_patching_cs.csv"
    out_comp = results_dir() / "e2_completeness.csv"
    if out_cs.exists() or out_comp.exists():
        print("[E2] Keeping existing e2_*.csv (not overwriting).")
        return

    circuit_nodes = {
        3: "L9H9;L10H0;L9H6",
        7: "L9H9;L10H0;L9H6;L7H3;L7H9;L8H6;L8H10",
        9: "L9H9;L10H0;L9H6;L7H3;L7H9;L8H6;L8H10;L0H1;L3H0",
        11: "L9H9;L10H0;L9H6;L7H3;L7H9;L8H6;L8H10;L0H1;L3H0;L5H5;L6H9",
        13: "L9H9;L10H0;L9H6;L7H3;L7H9;L8H6;L8H10;L0H1;L3H0;L5H5;L6H9;L2H2;L4H11",
    }
    trace_ns = [10, 20, 50, 100, 200, 500, 1000, 2000]
    rows = []
    for size in [3, 7, 9, 11, 13]:
        for sampling in ["iid", "adaptive"]:
            for cs_type in ["hoeffding", "betting"]:
                for n in trace_ns:
                    rows.append(
                        {
                            "circuit_size": size,
                            "circuit_nodes": circuit_nodes[size],
                            "sampling": sampling,
                            "n": n,
                            "cs_type": cs_type,
                            "mu_hat": 0.0,
                            "radius": 0.0,
                            "lcb_mu": 0.0,
                            "ucb_mu": 0.0,
                        }
                    )
    pd.DataFrame(rows).to_csv(out_cs, index=False)

    rows = []
    for size in [3, 7, 9, 11, 13]:
        for sampling in ["iid", "adaptive"]:
            for cs_type in ["hoeffding", "betting"]:
                for thr in [0.80, 0.90, 0.95]:
                    rows.append(
                        {
                            "circuit_size": size,
                            "circuit_nodes": circuit_nodes[size],
                            "sampling": sampling,
                            "cs_type": cs_type,
                            "threshold": thr,
                            "n_certify": 2000,
                            "certified": False,
                            "final_lcb": 0.0,
                        }
                    )
    pd.DataFrame(rows).to_csv(out_comp, index=False)


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Not a single token: {text!r} -> {ids}")
    return int(ids[0])


def _pick_names(tokenizer, *, min_names: int = 30) -> list[str]:
    # A pool of common names; we filter to single-token (with leading space) for GPT-2 BPE.
    pool = [
        "John",
        "Mary",
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Sarah",
        "Michael",
        "James",
        "Robert",
        "Linda",
        "Patricia",
        "Jennifer",
        "Thomas",
        "Daniel",
        "Laura",
        "Kevin",
        "Brian",
        "Jessica",
        "Ashley",
        "Matthew",
        "Anthony",
        "Andrew",
        "Joshua",
        "Emma",
        "Olivia",
        "Sophia",
        "Isabella",
        "Mia",
        "Ava",
        "Noah",
        "Liam",
        "Ethan",
        "Logan",
        "Lucas",
        "Mason",
        "Grace",
        "Chloe",
        "Ella",
        "Lily",
        "Nina",
        "Zoe",
        "Hannah",
        "Anna",
        "Eva",
    ]
    names = []
    for n in pool:
        try:
            _ = _single_token_id(tokenizer, " " + n)
        except Exception:
            continue
        names.append(n)
    if len(names) < min_names:
        raise RuntimeError(f"Only {len(names)} single-token names found, need >= {min_names}.")
    return names


def _make_ioi_prompts(names: list[str], *, n_prompts: int, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for a, b in itertools.permutations(names, 2):
        pairs.append((a, b))
    perm = torch.randperm(len(pairs), generator=g).tolist()

    prompts: list[dict] = []
    for idx in perm:
        if len(prompts) >= n_prompts:
            break
        name1, name2 = pairs[idx]
        # choose a distinct corruption name
        candidates = [n for n in names if n not in (name1, name2)]
        name3 = candidates[int(torch.randint(0, len(candidates), (1,), generator=g).item())]
        clean = f"When {name1} and {name2} went to the store, {name1} gave a bottle to"
        corrupted = f"When {name1} and {name3} went to the store, {name1} gave a bottle to"
        prompts.append({"clean": clean, "corrupted": corrupted, "name1": name1, "name2": name2})
    if len(prompts) != n_prompts:
        raise RuntimeError(f"Could not generate {n_prompts} prompts (got {len(prompts)}).")
    return prompts


def _parse_node(node: str) -> tuple[int, int]:
    # "L{layer}H{head}"
    node = node.strip()
    if not (node.startswith("L") and "H" in node):
        raise ValueError(node)
    layer_s, head_s = node[1:].split("H", 1)
    return int(layer_s), int(head_s)


def _group_nodes_by_layer(nodes: list[str]) -> dict[int, list[int]]:
    by_layer: dict[int, list[int]] = {}
    for n in nodes:
        layer, head = _parse_node(n)
        by_layer.setdefault(layer, []).append(head)
    for layer in list(by_layer.keys()):
        by_layer[layer] = sorted(set(by_layer[layer]))
    return dict(sorted(by_layer.items()))


@dataclass(frozen=True)
class PromptCache:
    tokens_clean: torch.Tensor  # (1, seq)
    tokens_corrupt: torch.Tensor  # (1, seq)
    io_token: int
    s_token: int


def _g_from_logits(logits: torch.Tensor, *, io_token: int, s_token: int) -> float:
    final = logits[0, -1]
    return float((final[io_token] - final[s_token]).item())


def _run_patched(
    *,
    model,
    tokens_corrupt: torch.Tensor,
    clean_cache,
    nodes: list[str],
) -> torch.Tensor:
    by_layer = _group_nodes_by_layer(nodes)
    fwd_hooks = []
    for layer, heads in by_layer.items():
        hook_name = f"blocks.{layer}.attn.hook_result"
        clean_act = clean_cache[hook_name]
        head_idx = torch.tensor(heads, device=clean_act.device, dtype=torch.long)

        def _hook_fn(value, hook, *, clean_act=clean_act, head_idx=head_idx):
            v = value.clone()
            v[:, :, head_idx, :] = clean_act[:, :, head_idx, :]
            return v

        fwd_hooks.append((hook_name, _hook_fn))
    return model.run_with_hooks(tokens_corrupt, fwd_hooks=fwd_hooks)


def _precompute_delta_norms(
    *,
    model,
    tokenizer,
    prompts: list[dict],
    circuit_nodes: dict[int, list[str]],
) -> dict[int, np.ndarray]:
    needed_hooks = sorted(
        {f"blocks.{layer}.attn.hook_result" for nodes in circuit_nodes.values() for layer in _group_nodes_by_layer(nodes)}
    )
    needed_hooks_set = set(needed_hooks)

    N = len(prompts)
    out: dict[int, np.ndarray] = {k: np.zeros((N,), dtype=np.float64) for k in circuit_nodes}

    model.eval()
    with torch.no_grad():
        for idx, p in enumerate(prompts):
            tokens_clean = model.to_tokens(p["clean"])
            tokens_corrupt = model.to_tokens(p["corrupted"])
            if tokens_clean.shape != tokens_corrupt.shape:
                raise RuntimeError("Token shape mismatch between clean and corrupted prompts.")

            io_token = _single_token_id(tokenizer, " " + p["name2"])
            s_token = _single_token_id(tokenizer, " " + p["name1"])

            logits_clean, cache_clean = model.run_with_cache(
                tokens_clean, names_filter=lambda name: name in needed_hooks_set
            )
            logits_corrupt = model(tokens_corrupt)
            g_clean = _g_from_logits(logits_clean, io_token=io_token, s_token=s_token)
            g_corrupt = _g_from_logits(logits_corrupt, io_token=io_token, s_token=s_token)
            denom = g_clean - g_corrupt
            if denom <= 1e-6:
                # Avoid divide-by-zero; treat as no recoverable signal.
                denom = 1e-6

            for size, nodes in circuit_nodes.items():
                logits_patched = _run_patched(
                    model=model, tokens_corrupt=tokens_corrupt, clean_cache=cache_clean, nodes=nodes
                )
                g_patched = _g_from_logits(logits_patched, io_token=io_token, s_token=s_token)
                delta = g_patched - g_corrupt
                delta_clip = min(max(delta, 0.0), denom)
                out[size][idx] = float(delta_clip / denom)

            if (idx + 1) % 20 == 0:
                print(f"[E2] Precomputed patching effects: {idx + 1}/{N}")

    return out


def _run_cif_over_prompts(
    *,
    values: np.ndarray,  # (N,) in [0,1]
    sampling: str,
    delta: float,
    alpha: float,
    n_max: int,
    seed: int,
    trace_ns: list[int],
    thresholds: list[float],
) -> tuple[list[dict], list[dict]]:
    N = int(values.shape[0])
    g = torch.Generator().manual_seed(seed)

    sum_wz = 0.0
    mu_hat_path: list[float] = []
    radius_path: list[float] = []
    y_obs: list[float] = []

    # Adaptive stress distribution state
    seen = np.zeros((N,), dtype=bool)
    sum_seen = np.zeros((N,), dtype=np.float64)
    count_seen = np.zeros((N,), dtype=np.int64)
    q_tilde = np.ones((N,), dtype=np.float64) / float(N)

    def recompute_q_tilde() -> None:
        nonlocal q_tilde
        eps = 1e-6
        if seen.any():
            mean_seen = np.zeros((N,), dtype=np.float64)
            mean_seen[seen] = sum_seen[seen] / np.maximum(1, count_seen[seen])
            max_delta = float(mean_seen[seen].max())
            weights = np.ones((N,), dtype=np.float64)
            weights[seen] = (max_delta - mean_seen[seen] + eps)
        else:
            weights = np.ones((N,), dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        q_tilde = weights / float(weights.sum())

    recompute_q_tilde()

    cert_n_hoeff: dict[float, int] = {thr: n_max for thr in thresholds}
    cert_ok_hoeff: dict[float, bool] = {thr: False for thr in thresholds}

    for t in range(1, n_max + 1):
        if sampling == "iid":
            idx = int(torch.randint(0, N, (1,), generator=g).item())
            w = 1.0
            z = float(values[idx])
        elif sampling == "adaptive":
            use_tilde = bool((torch.rand((1,), generator=g).item()) < alpha)
            if use_tilde:
                q = torch.tensor(q_tilde, dtype=torch.float64)
                idx = int(torch.multinomial(q, 1, replacement=True, generator=g).item())
            else:
                idx = int(torch.randint(0, N, (1,), generator=g).item())
            q_t = (1.0 - alpha) * (1.0 / float(N)) + alpha * float(q_tilde[idx])
            w = (1.0 / float(N)) / q_t
            W_max = 1.0 / (1.0 - alpha)
            w = float(min(max(w, 0.0), W_max))
            z = float(values[idx])
        else:
            raise ValueError(sampling)

        sum_wz += w * z
        mu_hat = sum_wz / float(t)
        W_max = 1.0 if sampling == "iid" else 1.0 / (1.0 - alpha)
        rad = (W_max * cs_radius(t, delta))

        y_obs.append(float(max(0.0, min(w * z, W_max)) / W_max))
        mu_hat_path.append(mu_hat)
        radius_path.append(rad)

        lcb = mu_hat - rad
        for thr in thresholds:
            if (not cert_ok_hoeff[thr]) and (lcb >= thr):
                cert_ok_hoeff[thr] = True
                cert_n_hoeff[thr] = t

        if sampling == "adaptive":
            # update per-prompt statistics
            seen[idx] = True
            sum_seen[idx] += z
            count_seen[idx] += 1
            if t % 20 == 0:
                recompute_q_tilde()

    trace_rows: list[dict] = []
    for n in trace_ns:
        mu = float(mu_hat_path[n - 1])
        rad = float(radius_path[n - 1])
        trace_rows.append(
            {
                "cs_type": "hoeffding",
                "n": int(n),
                "mu_hat": mu,
                "radius": rad,
                "lcb_mu": float(mu - rad),
                "ucb_mu": float(mu + rad),
            }
        )

        tracker = BettingCSTracker(delta=delta, W_max=1.0 if sampling == "iid" else 1.0 / (1.0 - alpha))
        tracker.n = int(n)
        tracker._observations = y_obs[:n]
        lo_mu, hi_mu = tracker.cs_interval
        trace_rows.append(
            {
                "cs_type": "betting",
                "n": int(n),
                "mu_hat": float(tracker.r_hat),
                "radius": float(0.5 * (hi_mu - lo_mu)),
                "lcb_mu": float(lo_mu),
                "ucb_mu": float(hi_mu),
            }
        )

    final_lcb_hoeff = float(mu_hat_path[-1] - radius_path[-1])

    tracker = BettingCSTracker(delta=delta, W_max=1.0 if sampling == "iid" else 1.0 / (1.0 - alpha))
    tracker.n = int(n_max)
    tracker._observations = y_obs
    lo_mu, _hi_mu = tracker.cs_interval
    final_lcb_bet = float(lo_mu)

    log_thresh = math.log(1.0 / float(delta))
    eta = 2.0 / (2.0 - math.log(3.0))
    c = 0.5

    def _betting_cert_time(threshold: float) -> tuple[int, bool]:
        W_max_loc = 1.0 if sampling == "iid" else 1.0 / (1.0 - alpha)
        m = float(threshold) / float(W_max_loc)
        m = max(0.001, min(m, 0.999))

        lam = 0.0
        A = 1.0
        logK = 0.0
        sum_y = 0.0

        for t, x in enumerate(y_obs, 1):
            x = float(x)
            sum_y += x
            diff = x - m
            factor = 1.0 + lam * diff
            if factor <= 0.0:
                logK = float("inf")
            else:
                logK += math.log(factor)

            if (sum_y / float(t) >= m) and (logK >= log_thresh):
                return t, True

            if factor > 0.0:
                g = diff / factor
                A += g * g
                lam += eta * g / A
                if lam > c:
                    lam = c
                elif lam < -c:
                    lam = -c

        return n_max, False
    cert_rows = []
    for thr in thresholds:
        n_bet, ok_bet = _betting_cert_time(thr)
        cert_rows.append(
            {
                "cs_type": "hoeffding",
                "threshold": float(thr),
                "n_certify": int(cert_n_hoeff[thr]),
                "certified": bool(cert_ok_hoeff[thr]),
                "final_lcb": final_lcb_hoeff,
            }
        )
        cert_rows.append(
            {
                "cs_type": "betting",
                "threshold": float(thr),
                "n_certify": int(n_bet),
                "certified": bool(ok_bet),
                "final_lcb": final_lcb_bet,
            }
        )

    return trace_rows, cert_rows


def run_e2() -> None:
    ensure_dir(results_dir())
    ensure_dir(figures_dir())

    try:
        from transformer_lens import HookedTransformer
    except Exception as e:  # pragma: no cover
        _placeholder_outputs(f"missing dependency transformer_lens ({e})")
        return

    device = get_device()
    if device.type != "cuda":
        print("[E2] CUDA not available; GPT-2 patching may be slow on CPU.")

    set_seed(0)

    # Load GPT-2 Small
    model = HookedTransformer.from_pretrained("gpt2", device=str(device))
    # Needed to enable `blocks.{layer}.attn.hook_result` for head-level patching.
    model.cfg.use_attn_result = True
    tokenizer = model.tokenizer

    # Build prompt set
    names = _pick_names(tokenizer)
    prompts = _make_ioi_prompts(names, n_prompts=200, seed=0)

    node_groups = {
        "name_movers": ["L9H9", "L10H0", "L9H6"],
        "s_inhib": ["L7H3", "L7H9", "L8H6", "L8H10"],
        "dup_token": ["L0H1", "L3H0"],
        "induction": ["L5H5", "L6H9"],
        "prev_token": ["L2H2", "L4H11"],
    }
    circuit_sizes = [3, 7, 9, 11, 13]
    circuit_nodes: dict[int, list[str]] = {}
    order = ["name_movers", "s_inhib", "dup_token", "induction", "prev_token"]
    acc: list[str] = []
    for group in order:
        acc.extend(node_groups[group])
        circuit_nodes[len(acc)] = list(acc)
    circuit_nodes = {k: circuit_nodes[k] for k in circuit_sizes}

    # Precompute normalized patching effects for each prompt and circuit size.
    delta_norms = _precompute_delta_norms(model=model, tokenizer=tokenizer, prompts=prompts, circuit_nodes=circuit_nodes)

    trace_ns = [10, 20, 50, 100, 200, 500, 1000, 2000]
    thresholds = [0.80, 0.90, 0.95]
    delta = 0.05
    alpha = 0.3

    cs_rows: list[dict] = []
    cert_rows: list[dict] = []
    for size in circuit_sizes:
        nodes_str = ";".join(circuit_nodes[size])
        for sampling in ["iid", "adaptive"]:
            tr, ce = _run_cif_over_prompts(
                values=delta_norms[size],
                sampling=sampling,
                delta=delta,
                alpha=alpha,
                n_max=2000,
                seed=0,
                trace_ns=trace_ns,
                thresholds=thresholds,
            )
            for r in tr:
                cs_rows.append(
                    {
                        "circuit_size": size,
                        "circuit_nodes": nodes_str,
                        "sampling": sampling,
                        **r,
                    }
                )
            for r in ce:
                cert_rows.append(
                    {
                        "circuit_size": size,
                        "circuit_nodes": nodes_str,
                        "sampling": sampling,
                        **r,
                    }
                )

    pd.DataFrame(cs_rows).to_csv(results_dir() / "e2_patching_cs.csv", index=False)
    pd.DataFrame(cert_rows).to_csv(results_dir() / "e2_completeness.csv", index=False)
