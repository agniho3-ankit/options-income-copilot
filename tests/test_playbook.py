import copy

import pytest
import yaml

from options_copilot.playbook import PlaybookError, load_playbook, validate


@pytest.fixture
def raw(repo_root):
    return yaml.safe_load((repo_root / "playbook.yaml").read_text())


def write(tmp_path, data):
    path = tmp_path / "playbook.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def errors_for(data):
    with pytest.raises(PlaybookError) as exc:
        validate(data)
    return str(exc.value)


def test_committed_playbook_is_valid(repo_root):
    pb = load_playbook(repo_root / "playbook.yaml")
    assert pb.version >= 1
    assert len(pb.sha256) == 64
    assert pb["entry.pop_min"] == 0.65


def test_decided_thresholds_are_encoded(repo_root):
    pb = load_playbook(repo_root / "playbook.yaml")
    assert pb["entry.position_risk_pct.block"] == 5
    assert pb["entry.portfolio_heat_pct.block"] == 30
    assert pb["entry.earnings.block_within_days"] == 7
    assert pb["entry.earnings.expiry_must_precede"] is True
    assert pb["management.r3_stop_loss.loss_credit_multiple"] == 1.0
    assert pb["management.r7_ex_dividend.leg"] == "short_call_itm"
    assert pb["builder.wing_width_pct"] == 3.0


def test_tiers_and_enterability(repo_root):
    pb = load_playbook(repo_root / "playbook.yaml")
    assert pb.tier_of("mrvl") == "tier_1"
    assert pb.tier_of("MSFT") == "tier_2"
    assert pb.tier_of("SCHW") == "watch"
    assert pb.tier_of("TICKER_A") is None
    assert pb.is_enterable("BAC") and not pb.is_enterable("AAPL") and not pb.is_enterable("TICKER_A")


def test_playbook_is_read_only(repo_root):
    pb = load_playbook(repo_root / "playbook.yaml")
    with pytest.raises(TypeError):
        pb.tree["entry"]["pop_min"] = 0.10
    with pytest.raises(AttributeError):
        pb.universe = None  # frozen dataclass
    assert isinstance(pb["universe.tier_1"], tuple)


def test_sha_changes_when_a_threshold_changes(tmp_path, raw, repo_root):
    original = load_playbook(repo_root / "playbook.yaml")
    raw["entry"]["pop_min"] = 0.70
    assert load_playbook(write(tmp_path, raw)).sha256 != original.sha256


def test_unknown_key_is_rejected_as_likely_typo(raw):
    raw["entry"]["pop_minimum"] = 0.65
    assert "entry.pop_minimum: unknown key" in errors_for(raw)


def test_missing_key_is_rejected(raw):
    del raw["management"]["r3_stop_loss"]["loss_credit_multiple"]
    assert "management.r3_stop_loss.loss_credit_multiple: missing" in errors_for(raw)


@pytest.mark.parametrize("path,value,fragment", [
    (("entry", "pop_min"), "0.65", "expected"),
    (("entry", "pop_min"), True, "boolean"),
    (("entry", "pop_min"), 1.2, "above the maximum"),
    (("entry", "dte_min"), 30.5, "expected int"),
    (("entry", "open_interest_min"), -1, "below the minimum"),
    (("entry", "pop_method"), "delta", "not one of"),
    (("universe", "tier_1"), ["MRVL", ""], "non-empty string"),
])
def test_bad_leaf_values_are_rejected(raw, path, value, fragment):
    section, key = path
    raw[section][key] = value
    assert fragment in errors_for(raw)


@pytest.mark.parametrize("mutate,fragment", [
    (lambda r: r["entry"]["position_risk_pct"].update(warn=5), "position_risk_pct.warn"),
    (lambda r: r["entry"]["portfolio_heat_pct"].update(warn=31), "portfolio_heat_pct.warn"),
    (lambda r: r["builder"].update(short_delta_min=0.30), "short_delta_min"),
    (lambda r: r["builder"].update(condor_short_delta_min=0.25), "condor_short_delta_min"),
    (lambda r: r["entry"].update(dte_target=61), "dte_target"),
    (lambda r: r["management"]["r2_roll"].update(dte_below=30), "r2_roll.dte_below"),
    (lambda r: r["entry"]["earnings"].update(block_within_days=30), "block_within_days"),
    (lambda r: r["screener"]["quality_weights"].update(trend=3.0), "must sum to 10"),
    (lambda r: r["universe"]["watch"].append("MRVL"), "already listed"),
    (lambda r: r["universe"]["tier_2"].append("schw2"), "upper case"),
])
def test_inconsistent_values_are_rejected(raw, mutate, fragment):
    mutate(raw)
    assert fragment in errors_for(raw)


def test_boundary_equal_dte_values_are_allowed(raw):
    raw["entry"].update(dte_min=45, dte_target=45, dte_max=45)
    validate(copy.deepcopy(raw))


def test_every_error_is_reported_at_once(raw):
    raw["entry"]["pop_min"] = 5
    raw["entry"]["open_interest_min"] = "many"
    message = errors_for(raw)
    assert "entry.pop_min" in message and "entry.open_interest_min" in message


def test_unreadable_or_malformed_files_raise_playbook_error(tmp_path):
    with pytest.raises(PlaybookError):
        load_playbook(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("entry: [unclosed")
    with pytest.raises(PlaybookError):
        load_playbook(bad)
    bad.write_text("- just\n- a list\n")
    with pytest.raises(PlaybookError, match="mapping"):
        load_playbook(bad)
