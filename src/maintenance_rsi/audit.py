"""Repository integrity audit for 部分空間正則化PCA／先行シグナル予測λ."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import IntEnum
import json
from pathlib import Path
import re


MAINTENANCE_RSI_VERSION = "maintenance-rsi-v1"


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: Severity
    message: str
    path: str

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["severity"] = self.severity.name
        return row


@dataclass(frozen=True)
class AuditReport:
    version: str
    generated_at_utc: str
    status: str
    findings: tuple[AuditFinding, ...]
    checked_controls: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "generated_at_utc": self.generated_at_utc,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "checked_controls": list(self.checked_controls),
            "autonomous_source_edits": False,
            "autonomous_main_merge": False,
            "trading_authority": False,
            "recursive_rsi_autonomous_parameter_promotion": True,
            "recursive_rsi_autonomous_source_promotion": False,
        }


_PINNED_ACTION = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}(?:\s*#.*)?$"
)
_BANNED_QUALIFIED_CALLS = {
    ("pickle", "load"): "untrusted pickle deserialization is prohibited",
    ("pickle", "loads"): "untrusted pickle deserialization is prohibited",
    ("yaml", "load"): "unsafe YAML loading is prohibited",
    ("yaml", "unsafe_load"): "unsafe YAML loading is prohibited",
    ("yaml", "full_load"): "unsafe YAML loading is prohibited",
    ("marshal", "loads"): "untrusted marshal deserialization is prohibited",
}
_BANNED_METHOD_CALLS = {
    "bfill": "future-directed bfill is prohibited",
    "backfill": "future-directed backfill is prohibited",
}
_SECRET_PATTERNS = {
    "AWS_ACCESS_KEY": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GITHUB_TOKEN": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}"),
    "PRIVATE_KEY": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _class_init_defaults(path: Path, class_name: str) -> dict[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            init = next(
                (item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
                None,
            )
            if init is None:
                return {}
            positional = init.args.args[-len(init.args.defaults):] if init.args.defaults else []
            values = {arg.arg: _number(default) for arg, default in zip(positional, init.args.defaults)}
            values.update(
                {
                    arg.arg: _number(default)
                    for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults)
                    if default is not None
                }
            )
            return {key: value for key, value in values.items() if value is not None}
    return {}


def _check_core(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    model_path = root / "src/leading_signal_lambda/model.py"
    paper_path = root / "src/leading_signal_lambda/paper_pca_sub.py"
    recursive_path = root / "src/leading_signal_lambda/recursive_self_improvement.py"
    runtime_path = root / "src/leading_signal_lambda/recursive_runtime.py"
    for path in (model_path, paper_path, recursive_path, runtime_path):
        if not path.is_file():
            findings.append(AuditFinding("CORE_FILE_MISSING", Severity.CRITICAL, "required core file is missing", str(path)))
    if findings:
        return findings

    defaults = _class_init_defaults(model_path, "LeadingLambdaClassifier")
    required = {"lambda_reg": 0.10, "variance_target": 0.90, "min_samples": 60.0}
    for name, expected in required.items():
        if defaults.get(name) != expected:
            findings.append(
                AuditFinding(
                    "CORE_INVARIANT_CHANGED",
                    Severity.CRITICAL,
                    f"LeadingLambdaClassifier {name} must remain {expected}",
                    str(model_path),
                )
            )

    paper_tree = ast.parse(paper_path.read_text(encoding="utf-8"), filename=str(paper_path))
    constants = {
        node.targets[0].id: _number(node.value)
        for node in paper_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    if constants.get("PAPER_LAMBDA_PRIOR") != 0.90:
        findings.append(
            AuditFinding(
                "PAPER_LAMBDA_CHANGED",
                Severity.CRITICAL,
                "PAPER_LAMBDA_PRIOR must remain 0.90",
                str(paper_path),
            )
        )
    recursive_text = recursive_path.read_text(encoding="utf-8")
    for required_guard in (
        '"autonomous_source_edits": False',
        '"autonomous_main_merge": False',
        '"trading_authority": False',
        '"human_approval_required": True',
    ):
        if required_guard not in recursive_text:
            findings.append(
                AuditFinding(
                    "RECURSIVE_RSI_GUARD_REMOVED",
                    Severity.CRITICAL,
                    f"required recursive RSI guard is missing: {required_guard}",
                    str(recursive_path),
                )
            )
    runtime_text = runtime_path.read_text(encoding="utf-8")
    for required_runtime_guard in (
        "MarketRecursiveImprovementGate",
        "baseline forecast changed after recursive RSI freeze",
        "candidate forecast changed after recursive RSI freeze",
        '"parameter_promotion_applied": applied',
    ):
        if required_runtime_guard not in runtime_text:
            findings.append(
                AuditFinding(
                    "RECURSIVE_RSI_RUNTIME_GUARD_REMOVED",
                    Severity.CRITICAL,
                    f"required recursive RSI runtime guard is missing: {required_runtime_guard}",
                    str(runtime_path),
                )
            )
    return findings


def _dangerous_call_findings(tree: ast.AST, path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in _BANNED_METHOD_CALLS:
                findings.append(
                    AuditFinding(
                        "DANGEROUS_SOURCE_PATTERN",
                        Severity.HIGH,
                        _BANNED_METHOD_CALLS[func.attr],
                        str(path),
                    )
                )
            if isinstance(func.value, ast.Name):
                message = _BANNED_QUALIFIED_CALLS.get((func.value.id, func.attr))
                if message is not None:
                    findings.append(AuditFinding("DANGEROUS_SOURCE_PATTERN", Severity.HIGH, message, str(path)))
        for keyword in node.keywords:
            if (
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                findings.append(
                    AuditFinding(
                        "DANGEROUS_SOURCE_PATTERN",
                        Severity.HIGH,
                        "shell=True is prohibited in prediction code",
                        str(path),
                    )
                )
    return findings


def _check_prediction_sources(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    source_roots = (root / "src/leading_signal_lambda", root / "src/maintenance_rsi")
    checked_paths = {
        path for source_root in source_roots for path in source_root.rglob("*.py")
    }
    for path in sorted(checked_paths):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            findings.append(AuditFinding("PYTHON_SYNTAX_ERROR", Severity.CRITICAL, str(exc), str(path)))
            continue
        findings.extend(_dangerous_call_findings(tree, path))
    return findings


def _check_workflows(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    workflow_root = root / ".github/workflows"
    for path in sorted((*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        if "pull_request_target:" in text:
            findings.append(AuditFinding("PRIVILEGED_PR_TRIGGER", Severity.CRITICAL, "pull_request_target is prohibited", str(path)))
        if re.search(r"(?m)^permissions:\s*write-all\s*$", text):
            findings.append(AuditFinding("WRITE_ALL_PERMISSION", Severity.CRITICAL, "write-all workflow permission is prohibited", str(path)))
        for match in re.finditer(r"(?m)^\s*-?\s*uses:\s*([^\s]+(?:\s*#.*)?)$", text):
            action = match.group(1).strip()
            if action.startswith("./"):
                continue
            if not _PINNED_ACTION.match(action):
                findings.append(
                    AuditFinding(
                        "UNPINNED_ACTION",
                        Severity.HIGH,
                        f"third-party action must be pinned to a full commit SHA: {action}",
                        str(path),
                    )
                )
    return findings


def _check_secret_leaks(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    candidates = [
        *(root / "src").rglob("*.py"),
        *(root / "tests").rglob("*.py"),
        *(root / ".github").rglob("*.yml"),
        *(root / ".github").rglob("*.yaml"),
        root / "pyproject.toml",
        root / "README.md",
    ]
    for path in sorted({candidate for candidate in candidates if candidate.is_file()}):
        if path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for secret_type, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    AuditFinding(
                        "SECRET_LEAK",
                        Severity.CRITICAL,
                        f"high-confidence {secret_type} signature detected",
                        str(path),
                    )
                )
    return findings


def audit_repository(root: str | Path) -> AuditReport:
    repository = Path(root).resolve()
    findings = [
        *_check_core(repository),
        *_check_prediction_sources(repository),
        *_check_workflows(repository),
        *_check_secret_leaks(repository),
    ]
    highest = max((finding.severity for finding in findings), default=Severity.INFO)
    status = "BLOCK" if highest >= Severity.HIGH else "PASS"
    return AuditReport(
        version=MAINTENANCE_RSI_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        status=status,
        findings=tuple(findings),
        checked_controls=(
            "fixed_model_invariants",
            "python_syntax",
            "no_future_fill",
            "unsafe_deserialization",
            "workflow_least_privilege",
            "action_sha_pinning",
            "high_confidence_secret_scan",
            "recursive_rsi_future_only_parameter_promotion_gate",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run maintenance-only RSI integrity controls")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_repository(args.root)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    raise SystemExit(1 if report.status == "BLOCK" else 0)


if __name__ == "__main__":
    main()
