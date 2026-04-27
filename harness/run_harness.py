from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.fixtures import install_fixture
from harness.graders import grade_scenario, replay_artifact
from harness.models import HarnessReport, HarnessScenario, ScenarioArtifact
from trading_agents.core.config import Settings
from trading_agents.core.services import AppServices


DEFAULT_SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"


def load_scenarios(scenario_dir: Path = DEFAULT_SCENARIO_DIR) -> list[HarnessScenario]:
    scenarios: list[HarnessScenario] = []
    for path in sorted(scenario_dir.glob("*.json")):
        scenarios.append(HarnessScenario.model_validate_json(path.read_text(encoding="utf-8")))
    return scenarios


def build_harness_settings(base_dir: Path) -> Settings:
    data_dir = base_dir / "data"
    return Settings(
        env="test",
        data_dir=data_dir,
        db_path=data_dir / "trading.db",
        chroma_persist_dir=data_dir / "chroma",
        langgraph_checkpoint_path=data_dir / "langgraph-checkpoints.sqlite",
        drahmi_api_key="",
        marketaux_api_key="",
        secret_key="harness-secret",
        langsmith_tracing=False,
        alpaca_enabled=True,
        alpaca_api_key_id="",
        alpaca_api_secret_key="",
    )


def run_scenario(scenario: HarnessScenario) -> ScenarioArtifact:
    with TemporaryDirectory(prefix=f"harness-{scenario.name}-") as temp_dir:
        settings = build_harness_settings(Path(temp_dir))
        services = AppServices(settings)
        try:
            install_fixture(services, scenario.fixture)
            response = services.generate(scenario.request)
            initial_detail = services.export_signal_detail(response.request_id)
            final_detail = initial_detail
            if scenario.approval_decision == "approve" and initial_detail.get("order_approval_required"):
                services.approve(response.request_id)
                final_detail = services.export_signal_detail(response.request_id)
            elif scenario.approval_decision == "reject" and initial_detail.get("order_approval_required"):
                services.reject(response.request_id)
                final_detail = services.export_signal_detail(response.request_id)
            else:
                final_detail = services.export_signal_detail(response.request_id)
            events = services.stream_events(response.request_id)
            grade = grade_scenario(
                scenario,
                initial_detail=initial_detail,
                final_detail=final_detail,
                events=events,
            )
            return ScenarioArtifact(
                name=scenario.name,
                description=scenario.description,
                fixture=scenario.fixture,
                request_id=response.request_id,
                approval_decision=scenario.approval_decision,
                scenario=scenario,
                initial_detail=initial_detail,
                final_detail=final_detail,
                events=events,
                generated_at=datetime.now(timezone.utc),
                grade=grade,
            )
        finally:
            services.close()


def build_report(scenarios: list[HarnessScenario]) -> HarnessReport:
    artifacts = [run_scenario(scenario) for scenario in scenarios]
    passed_count = sum(1 for artifact in artifacts if artifact.grade.passed)
    return HarnessReport(
        mode="execute",
        generated_at=datetime.now(timezone.utc),
        scenario_count=len(artifacts),
        passed_count=passed_count,
        failed_count=len(artifacts) - passed_count,
        artifacts=artifacts,
    )


def replay_report(report_path: Path) -> HarnessReport:
    source = HarnessReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    artifacts: list[ScenarioArtifact] = []
    for artifact in source.artifacts:
        replayed = artifact.model_copy(update={"grade": replay_artifact(artifact)})
        artifacts.append(replayed)
    passed_count = sum(1 for artifact in artifacts if artifact.grade.passed)
    return HarnessReport(
        mode="replay",
        generated_at=datetime.now(timezone.utc),
        scenario_count=len(artifacts),
        passed_count=passed_count,
        failed_count=len(artifacts) - passed_count,
        artifacts=artifacts,
    )


def write_report(report: HarnessReport, output_path: Path | None = None) -> Path:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = output_path or (DEFAULT_REPORT_DIR / f"harness-report-{report.generated_at.strftime('%Y%m%dT%H%M%SZ')}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return target


def format_summary(report: HarnessReport) -> str:
    lines = [
        f"Mode: {report.mode}",
        f"Scenarios: {report.scenario_count}",
        f"Passed: {report.passed_count}",
        f"Failed: {report.failed_count}",
    ]
    for artifact in report.artifacts:
        status = "PASS" if artifact.grade.passed else "FAIL"
        lines.append(f"{status} {artifact.name}")
        for failure in artifact.grade.failures:
            lines.append(f"  - {failure}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Morocco trading-agent harness.")
    parser.add_argument("--scenario", default="all", help="Scenario name to run, or 'all'.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the saved JSON report.")
    parser.add_argument("--replay", type=Path, default=None, help="Replay a previously saved harness report.")
    args = parser.parse_args()

    if args.replay is not None:
        report = replay_report(args.replay)
        output_path = write_report(report, args.output)
        print(format_summary(report))
        print(f"Saved report: {output_path}")
        return 0 if report.failed_count == 0 else 1

    scenarios = load_scenarios()
    if args.scenario != "all":
        scenarios = [scenario for scenario in scenarios if scenario.name == args.scenario]
        if not scenarios:
            raise SystemExit(f"Scenario not found: {args.scenario}")
    report = build_report(scenarios)
    output_path = write_report(report, args.output)
    print(format_summary(report))
    print(f"Saved report: {output_path}")
    return 0 if report.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
