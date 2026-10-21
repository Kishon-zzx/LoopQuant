"""The paper-aligned LoopQuant pipeline: CLDC, then ILPA, then LCTC."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .cldc import CrossLoopDistributionCanonicalization
from .config import LoopQuantConfig
from .ilpa import InteractionAwareLoopwisePrecisionAllocation
from .lctc import LoopConditionedTaskCalibration
from .models import load_looped_transformer
from .quantization import QUANTIZATION_CONTEXT, rotate_ouro_backbone


@dataclass
class LoopQuantResult:
    model: Any
    allocation: list[int]
    cldc: dict[str, Any]
    ilpa: dict[str, Any]
    lctc: dict[str, Any]
    stage_order: tuple[str, ...] = ("CLDC", "ILPA", "LCTC")


class LoopQuantPipeline:
    """Run the three LoopQuant components in their paper-defined order."""

    def __init__(self, config: LoopQuantConfig, device: str = "cuda:0", logger=None):
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger("loopquant")
        self.cldc: CrossLoopDistributionCanonicalization | None = None
        self.ilpa: InteractionAwareLoopwisePrecisionAllocation | None = None
        self.lctc: LoopConditionedTaskCalibration | None = None

    def run_cldc(self, model, calibration_data: torch.Tensor) -> dict[str, Any]:
        """Construct and freeze the shared cross-loop quantized representation."""
        options = self.config.method["cldc"]
        if not options.get("enabled", True):
            raise ValueError("CLDC is a required LoopQuant component")
        self.logger.info("[CLDC] Cross-loop distribution canonicalization")
        self.cldc = CrossLoopDistributionCanonicalization(
            group_size=int(self.config.quantization["group_size"]),
            optimization_steps=int(options.get("optimization_steps", 80)),
            learning_rate=float(options.get("learning_rate", 1e-3)),
            rotation_seed=int(options.get("rotation_seed", 42)),
            statistics_sequences=int(options.get("statistics_sequences", 16)),
        )
        return self.cldc.run(model, calibration_data, self.device)

    def run_ilpa(self, model, calibration_data: torch.Tensor) -> dict[str, Any]:
        """Select loop precision under the configured average activation-bit budget."""
        if self.cldc is None:
            raise RuntimeError("CLDC must run before ILPA")
        options = self.config.method["ilpa"]
        if not options.get("enabled", True):
            raise ValueError("ILPA is a required LoopQuant component")
        setting = str(self.config.quantization["activation_setting"]).lower()
        if setting == "w4a4":
            candidates, average = (4,), 4.0
        else:
            candidates = tuple(int(bit) for bit in self.config.quantization["activation_candidates"])
            average = float(self.config.quantization["average_activation_bits"])
        self.logger.info("[ILPA] Interaction-aware loop-wise precision allocation")
        self.ilpa = InteractionAwareLoopwisePrecisionAllocation(
            num_loops=int(self.config.num_loops),
            candidate_bits=candidates,
            average_bits=average,
            interaction_shrinkage=float(options.get("interaction_shrinkage", 0.5)),
            selection_samples=int(options.get("selection_samples", 32)),
        )
        return self.ilpa.run(
            model,
            calibration_data,
            self.cldc.statistics,
            self.device,
        )

    def run_lctc(
        self,
        model,
        teacher_model,
        calibration_data: torch.Tensor,
        allocation: list[int],
    ) -> dict[str, Any]:
        """Calibrate and attach static loop/module clipping, gain, and bias."""
        if self.ilpa is None:
            raise RuntimeError("ILPA must run before LCTC")
        options = self.config.method["lctc"]
        if not options.get("enabled", True):
            raise ValueError("LCTC is a required LoopQuant component")
        self.logger.info("[LCTC] Loop-conditioned task calibration")
        rotate_ouro_backbone(
            teacher_model,
            seed=int(self.config.method["cldc"].get("rotation_seed", 42)),
        )
        self.lctc = LoopConditionedTaskCalibration(
            num_loops=int(self.config.num_loops),
            group_size=int(self.config.quantization["group_size"]),
            updates=int(options.get("updates", 192)),
            learning_rate=float(options.get("learning_rate", 1e-3)),
            task_objective_weight=float(options.get("task_objective_weight", 0.03)),
            seed=int(self.config.calibration["seed"]),
        )
        self.lctc.initialize(model, teacher_model, calibration_data, allocation, self.device)
        summary = self.lctc.calibrate(model, teacher_model, calibration_data, self.device)
        self.lctc.apply(model)
        return summary

    def run(
        self,
        model,
        tokenizer,
        calibration_data: torch.Tensor,
        teacher_model=None,
    ) -> LoopQuantResult:
        """Execute CLDC -> ILPA -> LCTC with one calibration dataset."""
        self.logger.info("[LoopQuant] Starting CLDC -> ILPA -> LCTC")
        cldc_summary = self.run_cldc(model, calibration_data)
        ilpa_summary = self.run_ilpa(model, calibration_data)
        allocation = list(ilpa_summary["allocation"])
        if teacher_model is None:
            model_path = getattr(model, "_loopquant_model_path", None)
            if model_path is None:
                raise ValueError("LCTC requires teacher_model or a model loaded by load_looped_transformer")
            teacher_model, _ = load_looped_transformer(
                model_path,
                model_family=self.config.model_family,
                device=self.device,
                num_loops=int(self.config.num_loops),
            )
        # load_looped_transformer() resets QUANTIZATION_CONTEXT to all-4; apply ILPA's allocation after it.
        QUANTIZATION_CONTEXT.set_allocation(allocation)
        QUANTIZATION_CONTEXT.enabled = True
        lctc_summary = self.run_lctc(
            model, teacher_model, calibration_data, allocation
        )
        self.logger.info("[LoopQuant] Completed CLDC -> ILPA -> LCTC")
        return LoopQuantResult(model, allocation, cldc_summary, ilpa_summary, lctc_summary)


def quantize_looped_transformer(
    model,
    tokenizer,
    calibration_data: torch.Tensor,
    config: LoopQuantConfig,
    *,
    device: str = "cuda:0",
    teacher_model=None,
) -> LoopQuantResult:
    """Convenience wrapper around :class:`LoopQuantPipeline`."""
    return LoopQuantPipeline(config, device=device).run(
        model, tokenizer, calibration_data, teacher_model=teacher_model
    )
