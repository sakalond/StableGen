import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


INSTALLER_PATH = Path(__file__).parents[1] / "stablegen" / "setup" / "installer.py"
SPEC = importlib.util.spec_from_file_location("stablegen_setup_installer", INSTALLER_PATH)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class GitDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.original_git_exe = installer.GIT_EXE
        installer.GIT_EXE = "git"

    def tearDown(self):
        installer.GIT_EXE = self.original_git_exe

    def test_finds_linuxbrew_git_when_path_lookup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            git = Path(temp_dir) / ".linuxbrew" / "bin" / "git"
            git.parent.mkdir(parents=True)
            git.touch()
            with patch.object(installer.sys, "platform", "linux"), \
                    patch.object(Path, "home", return_value=Path(temp_dir)), \
                    patch.object(installer.shutil, "which", return_value=None):
                self.assertEqual(installer.git_executable(), str(git))

    def test_prefers_git_found_on_path(self):
        with patch.object(installer.sys, "platform", "linux"), \
                patch.object(installer.shutil, "which", return_value="/usr/bin/git") as which:
            self.assertEqual(installer.git_executable(), "/usr/bin/git")
            which.assert_called_once_with("git")

    def test_does_not_use_linuxbrew_fallback_off_linux(self):
        with patch.object(installer.sys, "platform", "darwin"), \
                patch.object(installer.shutil, "which", return_value=None):
            self.assertIsNone(installer.git_executable())


if __name__ == "__main__":
    unittest.main()
