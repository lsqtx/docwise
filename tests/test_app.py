"""使用 Streamlit AppTest 做界面冒烟测试，不需要浏览器和 API Key。"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppSmokeTests(unittest.TestCase):
    def test_initial_render(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.radio), 1)
        self.assertTrue(app.text_input(key="base_url_input").value)

    def test_switch_to_ollama(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        app.selectbox(key="provider").select("Ollama 本地模型").run()
        self.assertFalse(app.exception)
        self.assertEqual(
            app.text_input(key="model_input").value, "qwen2.5:7b"
        )
        self.assertEqual(
            app.text_input(key="base_url_input").value, "http://localhost:11434"
        )


if __name__ == "__main__":
    unittest.main()
