from pathlib import Path
from features.splitting import SEED, split_data
from bigdata.clean import validate_clean
from bigdata.common import stage_parser, write_parquet
from spark_session import get_spark


def split(frame, domain):
    return dict(
        zip(
            ("train", "val", "test"),
            split_data(validate_clean(frame, domain), domain, SEED),
        )
    )


def main():
    args = stage_parser(__doc__).parse_args()
    spark = get_spark()
    try:
        for name, part in split(spark.read.parquet(args.input), args.domain).items():
            write_parquet(part, f'{args.output.rstrip("/")}/{args.domain}_{name}')
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
