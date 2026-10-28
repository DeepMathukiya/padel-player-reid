"""Command line: python -m padel_reid <stage> [--config ...] [--set key=value ...]."""

from __future__ import annotations

import argparse
import warnings

import yaml

from padel_reid import api

STAGES = ["detect", "tracklets", "embed", "label", "assign", "render", "gtclip", "evaluate",
          "heldout", "analyze", "charts", "baselines", "run"]


def _parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(prog="padel_reid", description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set start_frame=0 end_frame=900")
    parser.add_argument("--assignment", help="render: parquet with player labels to draw")
    parser.add_argument("--output", default="padel_player_ids.mp4", help="render: output file name")
    parser.add_argument("--start", type=int, help="render: first frame")
    parser.add_argument("--end", type=int, help="render: end frame (exclusive)")
    parser.add_argument("--gt", help="evaluate: MOT-format ground-truth file")
    args = parser.parse_args(argv)

    cfg = api.load_config(args.config, _parse_overrides(args.set))
    if args.stage == "detect":
        api.run_detection(cfg)
    elif args.stage == "tracklets":
        print(api.run_tracklets(cfg).describe())
    elif args.stage == "embed":
        api.run_embedding(cfg)
    elif args.stage == "label":
        print("Thumbnail sheets written to", api.export_labeling_sheets(cfg))
    elif args.stage == "assign":
        api.run_assignment(cfg)
    elif args.stage == "render":
        print("Video written to", api.run_render(cfg, args.assignment, args.output, args.start, args.end))
    elif args.stage == "gtclip":
        name = args.output.removesuffix(".mp4") if args.output != "padel_player_ids.mp4" else f"clip_{args.start}"
        print("Clip + pre-annotation written to", api.export_ground_truth_clip(cfg, args.start, args.end, name))
    elif args.stage == "evaluate":
        print(api.run_evaluation(cfg, args.gt).to_string())
    elif args.stage == "heldout":
        print(api.run_heldout_evaluation(cfg).round(4).to_string())
    elif args.stage == "charts":
        for path in api.write_report_charts(cfg):
            print("chart:", path)
    elif args.stage == "analyze":
        print("Analysis written to", api.run_analysis(cfg))
    elif args.stage == "baselines":
        print(api.run_baselines(cfg))
    elif args.stage == "run":
        api.run_pipeline(cfg)


if __name__ == "__main__":
    main()
