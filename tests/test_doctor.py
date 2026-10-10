import pytest

from cammy.doctor import Findings, diagnose, is_cammy_command, report


def levels(findings):
    return [level for level, _ in diagnose(findings)]


def text(findings):
    return " ".join(message for _, message in diagnose(findings))


def healthy(**changes):
    base = dict(module_loaded=True, device_exists=True, card_name="Cammy", exclusive_caps=True, has_capture_cap=True,
                format=(1280, 720), cammy_running=True, holders=["cammy-gui"], browsers=["Brave"], pipewire="unknown")
    base.update(changes)
    return Findings(**base)


def test_a_healthy_setup_has_no_problems():
    f = healthy()
    assert "problem" not in levels(f)
    assert "Streaming" in text(f) or "streaming" in text(f)
    assert "reload" in text(f)                                   # always reminds that apps read the list at page load


def test_no_module_is_the_first_and_only_problem():
    result = diagnose(Findings(module_loaded=False))
    assert [level for level, _ in result] == ["problem", "note"]
    assert "cammy-setup-camera" in result[1][1]


def test_a_missing_device_names_other_loopbacks_if_any():
    f = Findings(module_loaded=True, device_exists=False, other_loopbacks=["/dev/video11 (Other)"])
    assert levels(f)[1] == "problem" and "/dev/video11" in text(f) and "--virtual-device" in text(f)


def test_nothing_streaming_explains_why_meet_cannot_see_it():
    f = healthy(has_capture_cap=False, format=None, cammy_running=False, holders=[])
    message = text(f)
    assert "problem" in levels(f) and "Nothing is sending video" in message
    assert "Virtual camera box" in message and "reload" in message


def test_a_running_cammy_with_the_camera_off_is_called_out():
    assert "switched off or failed" in text(healthy(has_capture_cap=False, format=None))
    assert "switched off or failed" not in text(healthy(has_capture_cap=False, format=None, cammy_running=False))


def test_exclusive_caps_off_is_a_problem_for_chromium_browsers():
    f = healthy(exclusive_caps=False)
    assert "problem" in levels(f) and "exclusive_caps=1" in text(f)


def test_pipewire_without_a_source_is_the_firefox_case():
    f = healthy(pipewire="no-source", browsers=["Firefox"])
    message = text(f)
    assert "problem" in levels(f) and "PipeWire has the device but no camera source" in message
    assert "systemctl --user restart wireplumber" in message and "Chromium-based browser" in message and "Firefox" in message


def test_a_browser_that_has_the_camera_open_proves_it_can_see_it_whatever_pipewire_says():
    f = healthy(pipewire="no-source", browsers=["Firefox"], holders=["firefox-bin", "python"], holding_browsers=["Firefox"])
    message = text(f)
    assert "problem" not in levels(f)
    assert "Firefox already has the camera open" in message and "reads the device directly" in message
    assert "wireplumber" not in message                               # no needless advice to restart anything


def test_pipewire_with_a_source_is_fine():
    f = healthy(pipewire="source", browsers=["Firefox"])
    assert "problem" not in levels(f) and "PipeWire exposes a camera source" in text(f)


def test_firefox_with_no_pipewire_tools_gets_a_hint_not_a_verdict():
    f = healthy(pipewire="unavailable", browsers=["Firefox"])
    assert "problem" not in levels(f) and "pw-dump" in text(f)


def test_holders_are_listed_once():
    assert "brave, cammy-gui" in text(healthy(holders=["cammy-gui", "brave", "brave"]))


def test_the_report_counts_problems_and_the_exit_status_follows():
    good, bad = report(healthy()), report(healthy(has_capture_cap=False))
    assert good[1] == 0 and "Everything looks fine." in good[0]
    assert bad[1] == 1 and "1 problem found." in bad[0]
    assert report(Findings(module_loaded=False))[1] == 1


@pytest.mark.parametrize("argv,expected", [
    (["/home/me/.local/bin/cammy-gui"], True),
    (["/home/me/.local/bin/cammy", "--virtual-camera"], True),
    (["/usr/bin/python3", "/home/me/.local/bin/cammy"], True),
    (["python", "-m", "cammy"], True),
    (["python", "-m", "cammy.gui"], True),
    (["python", "-m", "cammy.doctor"], False),                    # this very tool
    (["/home/me/.local/bin/cammy-doctor"], False),
    (["/usr/bin/konsole", "--workdir", "/home/me/cammy"], False),   # a folder that happens to be called cammy
    (["/bin/bash", "-c", "cd cammy && ls"], False),
    ([], False),
])
def test_only_real_cammy_commands_count_as_running(argv, expected):
    assert is_cammy_command(argv) is expected


def test_the_command_runs_on_this_machine_without_crashing(capsys):
    from cammy import doctor
    code = doctor.main(["--device", "/dev/video99"])               # a device that does not exist: a clean verdict, no traceback
    out = capsys.readouterr().out
    assert code == 1 and "Cammy virtual camera check for /dev/video99" in out
