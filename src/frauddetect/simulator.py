"""Threshold simulator core: everything the demo needs, from aggregated histograms only.

The demo must not ship BAF rows (CC BY-NC-ND 4.0), so the experiment exports counts of
applications per (score bin, label, age group). From those counts any threshold's rates,
fairness and three-tier cost can be recomputed exactly up to the bin resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frauddetect.decision.three_tier import TierCosts

GROUPS = ("younger", "older")


@dataclass
class ScoreHistogram:
    """``counts[label][group]`` holds application counts per bin (bins ordered by score)."""

    edges: np.ndarray  # n_bins + 1 bin edges in score space (lower edges are thresholds)
    counts: dict[int, dict[str, np.ndarray]]
    model: str
    period: str

    @classmethod
    def from_scores(
        cls, y, scores, groups, *, n_bins: int = 1000, model: str = "", period: str = ""
    ) -> ScoreHistogram:
        """Quantile bins, so thresholds can be placed at fine alert rates everywhere."""
        y = np.asarray(y).astype(int)
        s = np.asarray(scores, dtype=float)
        g = np.asarray(groups)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
        counts = {
            label: {
                group: np.bincount(idx[(y == label) & (g == group)], minlength=len(edges) - 1)
                for group in GROUPS
            }
            for label in (0, 1)
        }
        return cls(edges=edges, counts=counts, model=model, period=period)

    # ----------------------------------------------------------------- persistence
    def to_json(self, path: str | Path) -> None:
        payload = {
            "model": self.model,
            "period": self.period,
            "edges": self.edges.tolist(),
            "counts": {
                str(label): {g: c.astype(int).tolist() for g, c in per_group.items()}
                for label, per_group in self.counts.items()
            },
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ScoreHistogram:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        counts = {
            int(label): {g: np.asarray(c) for g, c in per_group.items()}
            for label, per_group in payload["counts"].items()
        }
        return cls(np.asarray(payload["edges"]), counts, payload["model"], payload["period"])

    # ----------------------------------------------------------------- queries
    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def _at_or_above(self, label: int, group: str | None, bin_index: int) -> int:
        groups = GROUPS if group is None else (group,)
        return int(sum(self.counts[label][g][bin_index:].sum() for g in groups))

    def total(self, label: int | None = None, group: str | None = None) -> int:
        labels = (0, 1) if label is None else (label,)
        return sum(self._at_or_above(lab, group, 0) for lab in labels)

    def bin_for_alert_rate(self, rate: float) -> int:
        """First bin index whose upper tail holds at most ``rate`` of all applications."""
        per_bin = sum(self.counts[lab][g] for lab in (0, 1) for g in GROUPS)
        tail = np.cumsum(per_bin[::-1])[::-1] / per_bin.sum()
        candidates = np.flatnonzero(tail <= rate)
        return int(candidates[0]) if candidates.size else self.n_bins

    def rates(self, bin_index: int) -> dict[str, float]:
        """Rates when flagging every application in bins ``>= bin_index``."""
        out = {
            "alert_rate": (
                self._at_or_above(0, None, bin_index) + self._at_or_above(1, None, bin_index)
            )
            / max(self.total(), 1),
            "tpr": self._at_or_above(1, None, bin_index) / max(self.total(1), 1),
            "fpr": self._at_or_above(0, None, bin_index) / max(self.total(0), 1),
        }
        for g in GROUPS:
            out[f"fpr_{g}"] = self._at_or_above(0, g, bin_index) / max(self.total(0, g), 1)
            out[f"tpr_{g}"] = self._at_or_above(1, g, bin_index) / max(self.total(1, g), 1)
        high = max(out["fpr_younger"], out["fpr_older"])
        out["fpr_ratio"] = 1.0 if high == 0 else min(out["fpr_younger"], out["fpr_older"]) / high
        return out

    def three_tier(self, bin_t1: int, bin_t2: int, costs: TierCosts) -> dict[str, float]:
        """Tier counts and expected cost for thresholds at bins ``t1 <= t2``."""
        if bin_t1 > bin_t2:
            raise ValueError("t1 must not exceed t2")
        f_ge1, f_ge2 = self._at_or_above(1, None, bin_t1), self._at_or_above(1, None, bin_t2)
        l_ge1, l_ge2 = self._at_or_above(0, None, bin_t1), self._at_or_above(0, None, bin_t2)
        fraud = {1: self.total(1) - f_ge1, 2: f_ge1 - f_ge2, 3: f_ge2}
        legit = {1: self.total(0) - l_ge1, 2: l_ge1 - l_ge2, 3: l_ge2}
        fraud_loss = (
            fraud[1] * costs.loss_full
            + fraud[2] * (1 - costs.verify_catch) * costs.loss_limited
            + fraud[3] * (1 - costs.review_catch) * costs.loss_full
        )
        operations = (fraud[2] + legit[2]) * costs.cost_verify + (
            fraud[3] + legit[3]
        ) * costs.cost_review
        customer = legit[2] * costs.cost_friction + legit[3] * costs.cost_attrition
        n = max(self.total(), 1)
        return {
            "tier1_share": (fraud[1] + legit[1]) / n,
            "tier2_share": (fraud[2] + legit[2]) / n,
            "tier3_share": (fraud[3] + legit[3]) / n,
            "fraud_tier1": fraud[1],
            "fraud_tier2": fraud[2],
            "fraud_tier3": fraud[3],
            # expected frauds stopped: escalation alone stops none, verification and review do
            "fraud_caught": fraud[2] * costs.verify_catch + fraud[3] * costs.review_catch,
            "fraud_loss_per_app": fraud_loss / n,
            "operations_per_app": operations / n,
            "customer_per_app": customer / n,
            "cost_per_app": (fraud_loss + operations + customer) / n,
        }
