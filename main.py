from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent


class PipelineTarget(str, Enum):
    FULL = "full"
    DATA = "data"
    RESIDUALS = "residuals"
    CATBOOST = "catboost"
    NHITS = "nhits"
    PATCHTST = "patchtst"
    TFT = "tft"
    MODELS = "models"


class PipelineExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StepDefinition:
    name: str
    module_name: str
    function_name: str
    optional: bool = False


class PipelineApplication:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._steps = {
            PipelineTarget.DATA: StepDefinition(
                name="data",
                module_name="Utils.DataAnalysis",
                function_name="analyze_source_data",
            ),
            PipelineTarget.RESIDUALS: StepDefinition(
                name="residuals",
                module_name="Utils.DataAnalysis",
                function_name="analyze_residuals",
            ),
            PipelineTarget.CATBOOST: StepDefinition(
                name="catboost",
                module_name="Utils.CatBoost",
                function_name="run_feature_importance",
            ),
            PipelineTarget.NHITS: StepDefinition(
                name="nhits",
                module_name="Utils.NHits",
                function_name="run_nhits",
            ),
            PipelineTarget.PATCHTST: StepDefinition(
                name="patchtst",
                module_name="Utils.PatchTST",
                function_name="run_patchtst",
            ),
            PipelineTarget.TFT: StepDefinition(
                name="tft",
                module_name="Utils.TFT",
                function_name="run_tft",
            ),
        }

    def run(self, target: PipelineTarget) -> bool:
        os.chdir(self.project_root)

        if target is PipelineTarget.FULL:
            return self._run_full_pipeline()
        if target is PipelineTarget.MODELS:
            return self._run_models_pipeline()

        return self._run_step(self._steps[target])

    def _run_full_pipeline(self) -> bool:
        if not self._run_step(self._steps[PipelineTarget.DATA]):
            return False

        if not self._run_models_pipeline():
            return False

        residuals_step = StepDefinition(
            name=self._steps[PipelineTarget.RESIDUALS].name,
            module_name=self._steps[PipelineTarget.RESIDUALS].module_name,
            function_name=self._steps[PipelineTarget.RESIDUALS].function_name,
            optional=True,
        )
        self._run_step(residuals_step)
        return True

    def _run_models_pipeline(self) -> bool:
        steps = (
            self._steps[PipelineTarget.CATBOOST],
            self._steps[PipelineTarget.NHITS],
            self._steps[PipelineTarget.PATCHTST],
            self._steps[PipelineTarget.TFT],
        )
        return self._run_sequence(steps)

    def _run_sequence(self, steps: Sequence[StepDefinition]) -> bool:
        for step in steps:
            if not self._run_step(step):
                return False
        return True

    def _run_step(self, step: StepDefinition) -> bool:
        callable_object = self._load_callable(step)

        try:
            result = callable_object()
        except Exception as exc:
            if step.optional:
                return False
            raise PipelineExecutionError(f"{step.name}: {self._describe_exception(exc)}") from exc

        if result is False:
            if step.optional:
                return False
            raise PipelineExecutionError(f"{step.name}: returned false")

        return True

    def _load_callable(self, step: StepDefinition) -> Callable[[], object]:
        module = self._import_module(step.module_name)
        callable_object = getattr(module, step.function_name, None)

        if not callable(callable_object):
            raise PipelineExecutionError(
                f"{step.name}: callable '{step.function_name}' not found"
            )

        return callable_object

    @staticmethod
    def _import_module(module_name: str) -> ModuleType:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise PipelineExecutionError(f"{module_name}: import failed: {detail}") from exc

    @staticmethod
    def _describe_exception(exc: Exception) -> str:
        detail = str(exc).strip()
        if detail:
            return detail
        return exc.__class__.__name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        default=PipelineTarget.FULL.value,
        choices=[target.value for target in PipelineTarget],
        metavar="target",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    application = PipelineApplication(PROJECT_ROOT)

    try:
        is_successful = application.run(PipelineTarget(args.target))
    except PipelineExecutionError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    return 0 if is_successful else 1


if __name__ == "__main__":
    sys.exit(main())
