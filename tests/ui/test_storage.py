"""Local data folder: default location, user override, backups, session restore, data-folder switching."""
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from propsim import storage
from propsim.ui import worker as W
from propsim.ui.main_window import MainWindow


def test_default_location_is_appdata(monkeypatch):
    monkeypatch.delenv("PROPSIM_HOME", raising=False)
    if os.name == "nt":
        assert storage.platform_default() == Path(os.environ["APPDATA"]) / "PropSim"
    assert storage.resolve_home() == storage.platform_default()


def test_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPSIM_HOME", str(tmp_path / "env"))
    assert storage.resolve_home() == tmp_path / "env"
    assert storage.resolve_home(tmp_path / "mine") == tmp_path / "mine"          # the user's choice wins


def test_backup_rotation(tmp_path):
    home = storage.ensure_layout(tmp_path)
    f = tmp_path / "cases" / "a.psim.json"
    f.write_text("{}")
    for _ in range(25):
        assert storage.backup_file(f, home, keep=5) is not None
    assert len(list((home / "backups").glob("a.psim.json.*.bak"))) == 5
    assert storage.backup_file(tmp_path / "missing.json", home) is None


@pytest.fixture()
def win(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(home=tmp_path / "home", use_settings=False)
    w.show()
    app.processEvents()
    yield w
    w.close()
    W.shutdown()


def test_layout_backup_session_and_switch(win, tmp_path):
    home = tmp_path / "home"
    assert all((home / d).is_dir() for d in storage.SUBDIRS)
    win.select_model("ep.trades")
    case = home / "cases" / "c.psim.json"
    win.save_case(str(case))
    win.save_case(str(case))                                                    # overwrite -> backup
    assert list((home / "backups").glob("c.psim.json.*.bak"))
    win.select_model("ep.ion")                                                  # leaving ep.ion later -> session
    win.close()
    assert (home / "backups" / "last_session.psim.json").is_file()
    win.select_model("ep.trades")
    win.restore_session()
    assert win.ctrl.key == "ep.ion"
    (home / "runs" / "run_01.psim.h5").write_bytes(b"")                         # unreadable file is skipped
    new = tmp_path / "elsewhere"
    win.choose_data_folder(str(new), copy=True)
    assert win.home == new and (new / "cases" / "c.psim.json").is_file() and (new / "backups").is_dir()
