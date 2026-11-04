"""Read-only cylinder artifact diagnostics. No inference or threshold correction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .auxiliary.artifact_paths import artifact_file

KEYS = ("step_feature_thresholds", "spike_step_feature_thresholds",
        "offset_step_feature_thresholds", "detrended_residual_scale_raw")


def inspect_array(values, limit=50, allow_zero=False):
    values = np.asarray(values)
    finite = np.isfinite(values)
    good = values[finite]
    mask = ~finite | (values < 0 if allow_zero else values <= 0)
    indices = np.argwhere(mask)
    bad_count = int(np.count_nonzero(mask))
    return {
        "shape": values.shape, "dtype": str(values.dtype),
        "min": float(good.min()) if good.size else None,
        "max": float(good.max()) if good.size else None,
        "zero": int(np.count_nonzero(finite & (values == 0))),
        "negative": int(np.count_nonzero(finite & (values < 0))),
        "nonfinite": int(np.count_nonzero(~finite)),
        "bad_count": bad_count,
        "bad": [(tuple(int(v) for v in index), str(values[tuple(index)]))
                for index in indices[:limit]],
    }


def inspect_package(root, group=None, limit=50, show_arrays=False):
    root = Path(root)
    meta = json.loads((root / "golden_meta.json").read_text(encoding="utf-8-sig"))
    models = meta["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("모델 그룹이 없습니다")
    if group is not None and group not in models:
        raise ValueError("모델 그룹 없음: " + group)
    lines, issues = ["실린더 임계값 점검 (읽기 전용)", "모델 폴더: " + str(root)], 0
    for name, info in models.items():
        if group is not None and name != group:
            continue
        lines += ["", "=" * 72, "GROUP: " + name, "전체 임계값:"]
        mode = info.get("cylinder_score_mode", "legacy")
        required = {"threshold", "step_feature_thresholds"}
        if mode != "legacy":
            required.update(("offset_threshold", "offset_step_feature_thresholds", "detrended_residual_scale_raw"))
        if int(info.get("spike_topk", 0)) > 0:
            required.update(("spike_threshold", "spike_step_feature_thresholds"))
        for key in ("threshold", "spike_threshold", "offset_threshold"):
            value = info.get(key)
            lines.append("  %s: %s%s" % (key, value, " (현재 모드 필수)" if key in required else ""))
            if key in info or key in required:
                try:
                    valid = np.isfinite(float(value)) and float(value) >= 0
                except (TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    issues += 1
                    lines.append("  → 0 이상·유한 임계값 아님")
        try:
            steps = sorted(info.get("step_offsets", {}), key=int)
            sensors = meta.get("feature_names") or ["C1", "C2", "C_DIFF"]
            lines.append("행 스텝 순서: " + ", ".join(map(str, steps)))
            lines.append("열 센서 순서: " + ", ".join(map(str, sensors)))
            path = artifact_file(root, info["stats_path"])
            lines.append("NPZ: " + str(path))
            with np.load(path, allow_pickle=False) as data:
                lines.append("NPZ 전체 키: " + ", ".join(data.files))
                for key in KEYS:
                    lines.append("\n" + key)
                    if key not in data.files:
                        lines.append("  없음" + (" (현재 모드 필수)" if key in required else " (현재 모드 선택 항목)"))
                        issues += int(key in required)
                        continue
                    try:
                        values = data[key]
                        stats = inspect_array(values, limit, allow_zero=key.endswith("thresholds"))
                        for field in ("shape", "dtype", "min", "max", "zero", "negative", "nonfinite"):
                            lines.append("  %s: %s" % (field, stats[field]))
                        issues += stats["bad_count"]
                        # Threshold rows map to ProcessStep; residual scale rows are aligned timepoints.
                        is_cells = key.endswith("thresholds")
                        if is_cells and (values.ndim != 2 or values.shape != (len(steps), len(sensors))):
                            issues += 1
                            lines.append("  → step_offsets/feature_names와 배열 크기 불일치")
                        for index, value in stats["bad"]:
                            label = ""
                            if len(index) == 2:
                                row, col = index
                                label = (" · 스텝 " + str(steps[row]) if is_cells and row < len(steps)
                                         else " · 정렬 시점 " + str(row))
                                if col < len(sensors):
                                    label += " · 센서 " + str(sensors[col])
                            lines.append("  문제 위치 %s: %s%s" % (index, value, label))
                        if stats["bad_count"] > limit:
                            lines.append("  ... 나머지 %d개 생략" % (stats["bad_count"] - limit))
                        if show_arrays:
                            lines.append(np.array2string(values, threshold=values.size))
                    except (TypeError, ValueError, OSError) as exc:
                        issues += 1
                        lines.append("  배열 읽기/검사 실패: " + str(exc))
        except (TypeError, ValueError, KeyError, OSError) as exc:
            issues += 1
            lines.append("  그룹 검사 실패: " + str(exc))
    lines += ["", "문제 항목/값 수: %d" % issues,
              "임계값·scale이 정상이어도 원본 센서의 NaN/inf 등은 별도 확인해야 합니다.",
              "원본 모델·NPZ를 수정하거나 추론하지 않았습니다."]
    return "\n".join(lines), issues


def default_root():
    src = Path(__file__).resolve().parents[1]
    parents = [src / "보조모델/실린더", src.parent / "보조모델/실린더"]
    for parent in parents:
        base = parent / "DSV1"
        if (base / "golden_meta.json").is_file():
            return base
        matches = list(base.glob("*/golden_meta.json"))
        if len(matches) == 1:
            return matches[0].parent
        if (parent / "golden_meta.json").is_file():
            return parent
    return parents[0] / "DSV1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="golden_meta.json이 있는 폴더 (기본: DSV1 자동 탐색)")
    parser.add_argument("--group", help="특정 그룹만 검사")
    parser.add_argument("--limit", type=int, default=50, help="배열별 문제 위치 출력 한도")
    parser.add_argument("--show-arrays", action="store_true", help="검사 배열 전체 출력")
    parser.add_argument("--output", type=Path, help="UTF-8 텍스트 보고서 저장")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다")
    try:
        report, issues = inspect_package(args.root or default_root(), args.group, args.limit, args.show_arrays)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("점검 실패: " + str(exc))
        return 2
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
