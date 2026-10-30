import os
import pandas as pd

src_root_dir = "POC학습데이터_2026-07-15_추가데이터"
tgt_root_dir = "add_train"

# 열 순서 (전체 센서)
ALL_ORDER = [
    "Date",
    "Time",
    "ProcessStep",
    "h_0111_PV",
    "h_010_PV",
    "h_008_PV",
    "h_009_PV",
    "h_012_PV", 
    "p_002_PV",
    "h_0052_PV",
    "h_013_PV",
    "h_007_PV",
    "h_0072_PV",
    "h_0161_PV",
    "p_005_PV",
    "h_015_PV",
    "p_006_PV",
    "t_005_PV", 
    "h_005_PV",
    "t_002_PV", 
    "h_003_PV", 
    "h_004_PV", 
    "h_0162_PV", 
    "p_008_PV",
    "h_0053_PV", 
    "t_001_PV", 
    "h_001_PV", 
    "h_002_PV", 
    "p_007_PV", 
    "h_0211_PV", 
    "h_0212_PV", 
    "h_0213_PV", 
    "h_0214_PV", 
    "h_0221_PV", 
    "h_0222_PV",
    "h_0223_PV", 
    "h_0224_PV",
    "d_002_PV", 
    "d_005_PV",
    "d_003_PV",
    "d_001_PV", 
    "d_004_PV",
    "d_0061_PV", 
    "d_0062_PV", 
    "h_0181_PV", 
    "h_0182_PV",
    "h_0183_PV", 
    "h_0184_PV",
    "h_0191_PV",
    "h_0192_PV",
    "h_0193_PV",
    "h_0194_PV",
    "h_006_PV",
    "p_003_PV",
    "t_003_PV",
    "h_0141_PV",
    "p_004_PV",
    "t_004_PV",
    "s_001_Press",
    "s_002_Press",
    "s_003_Press",
]

# IO 센서 순서
IO_ORDER = [    "9000002", "9000003", "9000004", "9000005", "9000006", "9000007",
    "9000008", "9000009", "9000010", "9000011", "9000012", "9000013",
    # "9000015", "9000016", "9000017", "9000018",
    ]

# HTP가 아닌 센서들
NO_HTP = [
    "d_002_PV", 
    "d_005_PV",
    "d_003_PV",
    "d_001_PV", 
    "d_004_PV",
    "d_0061_PV", 
    "d_0062_PV", 
    "s_001_Press",
    "s_002_Press",
    "s_003_Press",
]


def convert_log_dataframe(df: pd.DataFrame, with_io: bool = True,
                          only_htp: bool = True) -> pd.DataFrame:
    """Convert one tab-delimited SEMES log dataframe to the training CSV schema."""
    if "ProcessStep" not in df.columns:
        raise ValueError("log 필수 컬럼 누락: ProcessStep")

    # ProcessStep 열에서 숫자가 아닌 데이터 행 전체 제거
    out = df[pd.to_numeric(df["ProcessStep"], errors="coerce").notna()].copy()

    order = ALL_ORDER + IO_ORDER if with_io else ALL_ORDER
    missing = [c for c in order if c not in out.columns]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise ValueError(f"log 필수 컬럼 누락({len(missing)}개): {preview}{suffix}")

    out = out[order]
    if only_htp:
        out = out.drop(columns=NO_HTP)
    # 마지막 열 이후에 tab이 있어서 더미 열 자동 생성 -> 삭제
    out = out.loc[:, ~out.columns.str.startswith("Unnamed")]
    return out


def convert_log_file(src_file, with_io: bool = True,
                     only_htp: bool = True) -> pd.DataFrame:
    df = pd.read_csv(src_file, sep="\t")
    return convert_log_dataframe(df, with_io=with_io, only_htp=only_htp)


def main() -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    # for WITH_IO in [True, False]:
    for WITH_IO in [True]:
        # for ONLY_HTP in [True, False]:
        for ONLY_HTP in [True]:

            # IO 유지 여부에 따른 디렉토리 설정
            tgt_dir = f"{tgt_root_dir}/converted_w_IO" if WITH_IO else f"{tgt_root_dir}/converted_wo_IO"

            # HTP만 할지 여부에 따른 디렉토리 설정
            tgt_dir = os.path.join(tgt_dir, "only_htp") if ONLY_HTP else os.path.join(tgt_dir, "all_sensors")
            for src_dir, _, src_filenames in tqdm(os.walk(src_root_dir)):
                for src_filename in src_filenames:
                    src_file = os.path.join(src_dir, src_filename)
                    df = convert_log_file(src_file, with_io=WITH_IO, only_htp=ONLY_HTP)

                    tgt_n = src_filename.split(".")[0]
                    tgt_name = "_336_" + tgt_n + ".csv"
                    # tgt_name = tgt_n + ".csv"

                    if tgt_n[-2:] == "OK":
                        tgt_file = os.path.join(tgt_dir, "OK")
                        os.makedirs(tgt_file, exist_ok=True)
                    elif tgt_n[-2:] == "NG":
                        tgt_file = os.path.join(tgt_dir, "NG")
                        os.makedirs(tgt_file, exist_ok=True)
                    else:
                        tgt_file = os.path.join(tgt_dir, "unknown_label")
                        os.makedirs(tgt_file, exist_ok=True)
                    tgt_file = os.path.join(tgt_file, tgt_name)
                    df.to_csv(tgt_file, sep=',', index=False)


if __name__ == "__main__":
    main()
