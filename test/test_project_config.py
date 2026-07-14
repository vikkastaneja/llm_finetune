"""Tests for the shared project config loader."""

import project_config


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.yaml")
    config = project_config.load_config(missing_path)
    assert config["held_out"] == {"size": 30, "seed": 42}
    assert config["severity_distribution"] == {"Healthy": 0.25, "Warning": 0.40, "Damaged": 0.35}


def test_real_config_file_has_expected_shape():
    # Checks structure, not specific values -- project_config.yaml is a live,
    # user-editable operational file (e.g. for one-off experiments with a
    # different held-out size or distribution), not a fixed test fixture.
    # Asserting exact values here would make legitimate config edits look
    # like a broken test.
    config = project_config.load_config("project_config.yaml")
    assert set(config["held_out"].keys()) == {"size", "seed"}
    assert set(config["severity_distribution"].keys()) == {"Healthy", "Warning", "Damaged"}
    assert set(config["total_records"].keys()) == {"n", "seed"}
    assert abs(sum(config["severity_distribution"].values()) - 1.0) < 0.01


def test_partial_override_does_not_drop_sibling_keys(tmp_path):
    # Regression test: a config that only overrides held_out.seed must not
    # silently lose held_out.size (a shallow-merge bug caught before shipping).
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text("held_out:\n  seed: 999\n")

    config = project_config.load_config(str(custom_path))
    assert config["held_out"]["seed"] == 999
    assert config["held_out"]["size"] == 30  # unchanged from fallback


def test_full_custom_config_overrides_everything(tmp_path):
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(
        "held_out:\n  size: 50\n  seed: 7\n"
        "severity_distribution:\n  Healthy: 0.5\n  Warning: 0.3\n  Damaged: 0.2\n"
    )

    config = project_config.load_config(str(custom_path))
    assert config["held_out"] == {"size": 50, "seed": 7}
    assert config["severity_distribution"] == {"Healthy": 0.5, "Warning": 0.3, "Damaged": 0.2}


def test_empty_config_file_falls_back_to_defaults(tmp_path):
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("")

    config = project_config.load_config(str(empty_path))
    assert config["held_out"] == {"size": 30, "seed": 42}
