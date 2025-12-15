import glob
import importlib
import os
import subprocess
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state


def _iter_name(iteration: int, compression_ratio: float, use_semantic_weights: bool, unbind_selection: bool, scale_expansion: bool, scale_expansion_mode: str) -> str:
    iter_name = f"ours_{iteration}"
    if compression_ratio < 1.0:
        suffix_parts = [f"comp{compression_ratio:.2f}".replace(".", "")]
        if use_semantic_weights:
            suffix_parts.append("semantic")
        if unbind_selection:
            suffix_parts.append("unbind")
        if scale_expansion:
            suffix_parts.append(f"scale_{scale_expansion_mode}")
        iter_name = f"{iter_name}_{'_'.join(suffix_parts)}"
    return iter_name


def _encode_video(input_dir: str, output_path: str, fps: int, codec: str, crf: int, preset: str) -> None:
    if not os.path.isdir(input_dir):
        return
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if len(frames) == 0:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(input_dir, "*.png"),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    render_mod = importlib.import_module("render")

    parser = ArgumentParser(description="4D rendering wrapper")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    def safe_add_argument(*args, **kwargs):
        for a in args:
            if isinstance(a, str) and a.startswith("-"):
                if a in parser._option_string_actions:
                    return None
        return parser.add_argument(*args, **kwargs)

    safe_add_argument("--iteration", default=-1, type=int)
    safe_add_argument("--skip_train", action="store_true")
    safe_add_argument("--skip_val", action="store_true")
    safe_add_argument("--skip_test", action="store_true")
    safe_add_argument("--quiet", action="store_true")
    safe_add_argument("--render_mesh", action="store_true")

    safe_add_argument("--compression_ratio", type=float, default=1.0)
    safe_add_argument("--use_semantic_weights", action="store_true")
    safe_add_argument("--unbind_selection", action="store_true")
    safe_add_argument("--scale_expansion", action="store_true")
    safe_add_argument("--scale_expansion_mode", type=str, default="volume", choices=["volume", "linear"])
    safe_add_argument("--use_gumbel", action="store_true")

    safe_add_argument("--interpolate_frames", action="store_true")
    safe_add_argument("--frames_per_timestep", type=int, default=1)

    safe_add_argument("--encode_video", action="store_true")
    safe_add_argument("--video_fps", type=int, default=25)
    safe_add_argument("--video_codec", type=str, default="libx264")
    safe_add_argument("--video_crf", type=int, default=18)
    safe_add_argument("--video_preset", type=str, default="medium")

    args = get_combined_args(parser)

    if hasattr(render_mod, "os") and hasattr(render_mod.os, "system"):
        render_mod.os.system = lambda *_a, **_kw: 0

    safe_state(args.quiet)

    render_mod.render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_val,
        args.skip_test,
        args.render_mesh,
        compression_ratio=args.compression_ratio,
        use_semantic_weights=args.use_semantic_weights,
        unbind_selection=args.unbind_selection,
        scale_expansion=args.scale_expansion,
        scale_expansion_mode=args.scale_expansion_mode,
        use_gumbel=args.use_gumbel,
        interpolate_frames=args.interpolate_frames,
        frames_per_timestep=args.frames_per_timestep,
    )

    if not args.encode_video:
        return

    set_names = []
    if not args.skip_train:
        set_names.append("train")
    if not args.skip_val:
        set_names.append("val")
    if not args.skip_test:
        set_names.append("test")

    select_camera_id = getattr(args, "select_camera_id", -1)
    if select_camera_id != -1:
        set_names = [f"{n}_{select_camera_id}" for n in set_names]

    iter_name = _iter_name(
        iteration=args.iteration,
        compression_ratio=args.compression_ratio,
        use_semantic_weights=args.use_semantic_weights,
        unbind_selection=args.unbind_selection,
        scale_expansion=args.scale_expansion,
        scale_expansion_mode=args.scale_expansion_mode,
    )

    for set_name in set_names:
        iter_dir = os.path.join(args.model_path, set_name, iter_name)
        _encode_video(
            os.path.join(iter_dir, "renders"),
            os.path.join(iter_dir, "renders_4d.mp4"),
            fps=args.video_fps,
            codec=args.video_codec,
            crf=args.video_crf,
            preset=args.video_preset,
        )
        _encode_video(
            os.path.join(iter_dir, "gt"),
            os.path.join(iter_dir, "gt_4d.mp4"),
            fps=args.video_fps,
            codec=args.video_codec,
            crf=args.video_crf,
            preset=args.video_preset,
        )
        if args.render_mesh:
            _encode_video(
                os.path.join(iter_dir, "renders_mesh"),
                os.path.join(iter_dir, "renders_mesh_4d.mp4"),
                fps=args.video_fps,
                codec=args.video_codec,
                crf=args.video_crf,
                preset=args.video_preset,
            )


if __name__ == "__main__":
    main()
