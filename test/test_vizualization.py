import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
from numpy.testing import assert_array_equal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch

from src.vizualization import plot_drr_enhancement, plot_results


class TestPlotDrrEnhancement(unittest.TestCase):
    def setUp(self):
        self.drr_body = torch.rand(1, 10, 10)
        self.drr_vessels = torch.rand(1, 10, 10)
        self.enhancement_factors = [1, 2, 3]
        self.cmap = "gray"
        self.vmax = 20

    @patch.object(plt, "subplots")
    @patch.object(plt, "show")
    def test_plot_drr_enhancement(self, mock_show, mock_subplots):
        mock_ax = MagicMock()
        mock_subplots.return_value = (None, [mock_ax] * len(self.enhancement_factors))

        plot_drr_enhancement(
            self.drr_body,
            self.drr_vessels,
            self.enhancement_factors,
            self.cmap,
        )

        # Convert calls to imshow into numpy arrays for comparison
        imshow_calls_args = [call.args[0] for call in mock_ax.imshow.call_args_list]
        imshow_calls_args_np = [
            arg.numpy() if hasattr(arg, "numpy") else arg for arg in imshow_calls_args
        ]

        for ef in self.enhancement_factors:
            expected_combined = (self.drr_body + ef * self.drr_vessels).squeeze()
            expected_combined_np = (
                expected_combined.numpy()
                if hasattr(expected_combined, "numpy")
                else expected_combined
            )

            # Check if any of the calls to imshow match the expected_combined_np
            found_match = any(
                np.array_equal(call_arg, expected_combined_np)
                for call_arg in imshow_calls_args_np
            )
            self.assertTrue(
                found_match, f"Expected call with enhancement factor {ef} not found."
            )


class TestPlotResults(unittest.TestCase):
    def setUp(self):
        self.drr_combined_low_enhancement = torch.rand(1, 10, 10)
        self.drr_combined_target = torch.rand(1, 10, 10)
        self.prediction = torch.rand(1, 10, 10)
        self.latent_representation = torch.rand(1, 10, 10)
        self.vmax = 20

    @patch.object(plt, "figure")
    @patch.object(plt, "tight_layout")
    def test_plot_results_creates_correct_number_of_subplots(
        self, mock_tight_layout, mock_figure
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        plot_results(
            self.drr_combined_low_enhancement,
            self.drr_combined_target,
            self.prediction,
            self.latent_representation,
            self.vmax,
        )

        self.assertEqual(mock_fig.add_subplot.call_count, 4)

    def test_plot_results_sets_titles_correctly(self):
        titles = ["DRR", "AI Enhanced", "Enhanced Target", "Latent Representation"]
        with patch.object(plt, "figure") as mock_figure, patch.object(
            plt, "tight_layout"
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax

            plot_results(
                self.drr_combined_low_enhancement,
                self.drr_combined_target,
                self.prediction,
                self.latent_representation,
                self.vmax,
            )

            title_calls = [call(title) for title in titles]
            mock_ax.set_title.assert_has_calls(title_calls, any_order=False)

    def test_plot_results_turns_axis_off(self):
        with patch.object(plt, "figure") as mock_figure, patch.object(
            plt, "tight_layout"
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax

            plot_results(
                self.drr_combined_low_enhancement,
                self.drr_combined_target,
                self.prediction,
                self.latent_representation,
                self.vmax,
            )

            self.assertEqual(mock_ax.axis.call_count, 4)
            mock_ax.axis.assert_called_with("off")


if __name__ == "__main__":
    unittest.main()
