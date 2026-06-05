import re
import unittest
from pathlib import Path


FIND_MACA = Path(__file__).parents[3] / "cmake" / "utils" / "FindMACA.cmake"


class FindMACACMakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = FIND_MACA.read_text(encoding="utf-8")

    def test_search_order_includes_maca_home_after_maca_path(self):
        maca_path_pos = self.content.index("$ENV{MACA_PATH}")
        maca_home_pos = self.content.index("$ENV{MACA_HOME}")
        default_pos = self.content.index("/opt/maca")

        self.assertLess(maca_path_pos, maca_home_pos)
        self.assertLess(maca_home_pos, default_pos)

    def test_public_outputs_are_reset_before_search(self):
        for variable in [
            "MACA_FOUND",
            "MACA_ROOT_DIR",
            "MACA_INCLUDE_DIRS",
            "MACA_MACAMCC_LIBRARY",
            "MACA_HCA_LIBRARY",
            "MACA_FLASHATTN_LIBRARY",
        ]:
            self.assertRegex(self.content, rf"unset\({re.escape(variable)}\)")

    def test_libraries_are_resolved_inside_selected_sdk(self):
        for variable in [
            "MACA_MACAMCC_LIBRARY",
            "MACA_HCA_LIBRARY",
            "MACA_FLASHATTN_LIBRARY",
        ]:
            self.assertRegex(
                self.content,
                rf"find_library\({variable} .* PATHS \${{__maca_sdk}}/lib NO_DEFAULT_PATH\)",
            )


if __name__ == "__main__":
    unittest.main()
