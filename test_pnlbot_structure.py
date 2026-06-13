import importlib
from pathlib import Path
import unittest


class PackageStructureTests(unittest.TestCase):
    def test_domain_modules_are_importable(self):
        module_names = [
            "pnlbot.config",
            "pnlbot.constants",
            "pnlbot.commands",
            "pnlbot.command_handlers",
            "pnlbot.config_commands",
            "pnlbot.http",
            "pnlbot.logging",
            "pnlbot.market_data",
            "pnlbot.messages",
            "pnlbot.models",
            "pnlbot.monitoring",
            "pnlbot.outages",
            "pnlbot.persistence",
            "pnlbot.runtime",
            "pnlbot.state",
            "pnlbot.system_info",
            "pnlbot.telegram",
            "pnlbot.time_utils",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_core_public_modules_expose_expected_symbols(self):
        from pnlbot.models import BotState
        from pnlbot.telegram import split_telegram_message

        self.assertIsNotNone(BotState)
        self.assertTrue(callable(split_telegram_message))

    def test_legacy_wrapper_file_is_removed(self):
        self.assertFalse(Path(__file__).with_name("PnLBot.py").exists())

    def test_main_module_is_starting_point(self):
        main_module = importlib.import_module("main")
        runtime = importlib.import_module("pnlbot.runtime")

        self.assertTrue(callable(main_module.main))
        self.assertIs(main_module.main, runtime.main)

    def test_command_router_lives_in_commands_module(self):
        commands = importlib.import_module("pnlbot.commands")
        command_handlers = importlib.import_module("pnlbot.command_handlers")
        config_commands = importlib.import_module("pnlbot.config_commands")

        self.assertTrue(callable(commands.check_telegram_commands))
        self.assertTrue(callable(command_handlers.handle_status_command))
        self.assertTrue(callable(config_commands.handle_config_command))

    def test_monitoring_lives_in_monitoring_module(self):
        monitoring = importlib.import_module("pnlbot.monitoring")

        self.assertTrue(callable(monitoring.monitor_loop))
        self.assertTrue(callable(monitoring.refresh_power_outages))
        self.assertTrue(callable(monitoring.start_system_monitor_worker))


if __name__ == "__main__":
    unittest.main()
