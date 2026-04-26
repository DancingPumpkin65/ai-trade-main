from pathlib import Path

from harness.run_harness import build_report, load_scenarios, replay_report, write_report


def test_harness_executes_all_scenarios_and_saves_report(tmp_path: Path):
    scenarios = load_scenarios()
    report = build_report(scenarios)

    assert report.scenario_count == 4
    assert report.failed_count == 0
    assert report.passed_count == 4

    report_path = write_report(report, tmp_path / "harness-report.json")
    assert report_path.exists()


def test_harness_replays_saved_report_without_live_execution(tmp_path: Path):
    scenario = next(item for item in load_scenarios() if item.name == "forced_buy_bias_refused")
    report = build_report([scenario])
    report_path = write_report(report, tmp_path / "saved-report.json")

    replayed = replay_report(report_path)
    assert replayed.mode == "replay"
    assert replayed.failed_count == 0
    assert replayed.artifacts[0].name == "forced_buy_bias_refused"
