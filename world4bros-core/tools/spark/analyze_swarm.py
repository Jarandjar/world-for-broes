#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


REQUIRED_COLUMNS = {
	"t", "type", "actorTeam", "actorKind", "weapon", "victimTeam", "victimKind", "distM", "dmg", "hpAfter"
}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Analyze swarm telemetry CSV with Spark")
	p.add_argument("--input", required=True, help="Path to swarm event CSV (single file or glob)")
	p.add_argument("--output", default="tools/spark/out", help="Output directory for aggregated parquet")
	p.add_argument("--app-name", default="w4b-swarm-telemetry", help="Spark app name")
	return p.parse_args()


def ensure_columns(cols: list[str]) -> None:
	missing = REQUIRED_COLUMNS - set(cols)
	if missing:
		raise SystemExit(f"Missing required telemetry columns: {', '.join(sorted(missing))}")


def main() -> None:
	args = parse_args()
	matched = glob.glob(args.input)
	if not matched:
		print(f"No telemetry files matched: {args.input}")
		print("Place CSV logs under tools/spark/input/ and rerun.")
		return

	spark = (
		SparkSession.builder
		.appName(args.app_name)
		.master("local[*]")
		.getOrCreate()
	)

	try:
		df = (
			spark.read
			.option("header", True)
			.option("inferSchema", True)
			.csv(args.input)
		)
		ensure_columns(df.columns)

		events = df.withColumn("distBand", F.when(F.col("distM") < 3, F.lit("0-3"))
							   .when(F.col("distM") < 6, F.lit("3-6"))
							   .when(F.col("distM") < 10, F.lit("6-10"))
							   .when(F.col("distM") < 15, F.lit("10-15"))
							   .when(F.col("distM") < 25, F.lit("15-25"))
							   .otherwise(F.lit("25+")))

		per_weapon = (
			events.groupBy("actorTeam", "weapon")
			.agg(
				F.sum(F.when(F.col("type") == "shot", 1).otherwise(0)).alias("shots"),
				F.sum(F.when(F.col("type") == "hit", 1).otherwise(0)).alias("hits"),
				F.sum(F.when(F.col("type") == "kill", 1).otherwise(0)).alias("kills"),
				F.sum(F.when(F.col("type").isin("hit", "kill"), F.col("dmg")).otherwise(0.0)).alias("dmg"),
				F.avg(F.when(F.col("type") == "kill", F.col("distM"))).alias("avgKillDist")
			)
			.withColumn("accuracy", F.when(F.col("shots") > 0, F.col("hits") / F.col("shots")).otherwise(F.lit(0.0)))
			.withColumn("dmgPerShot", F.when(F.col("shots") > 0, F.col("dmg") / F.col("shots")).otherwise(F.lit(0.0)))
		)

		per_band = (
			events.filter(F.col("type").isin("shot", "hit", "kill"))
			.groupBy("actorTeam", "weapon", "distBand")
			.agg(
				F.sum(F.when(F.col("type") == "shot", 1).otherwise(0)).alias("shots"),
				F.sum(F.when(F.col("type") == "hit", 1).otherwise(0)).alias("hits"),
				F.sum(F.when(F.col("type") == "kill", 1).otherwise(0)).alias("kills"),
				F.sum(F.when(F.col("type").isin("hit", "kill"), F.col("dmg")).otherwise(0.0)).alias("dmg")
			)
		)

		run_summary = (
			events.groupBy()
			.agg(
				F.sum(F.when(F.col("type") == "kill", 1).otherwise(0)).alias("kills"),
				F.sum(F.when(F.col("type") == "shot", 1).otherwise(0)).alias("shots"),
				F.sum(F.when(F.col("type").isin("hit", "kill"), F.col("dmg")).otherwise(0.0)).alias("dmg"),
				F.max("t").alias("maxTime")
			)
			.withColumn("killsPerMin", F.when(F.col("maxTime") > 0, F.col("kills") / (F.col("maxTime") / 60.0)).otherwise(F.lit(0.0)))
			.withColumn("accuracy", F.when(F.col("shots") > 0, F.lit(1.0) * F.col("kills") / F.col("shots")).otherwise(F.lit(0.0)))
		)

		out_root = Path(args.output)
		out_root.mkdir(parents=True, exist_ok=True)

		per_weapon.coalesce(1).write.mode("overwrite").parquet(str(out_root / "per_weapon"))
		per_band.coalesce(1).write.mode("overwrite").parquet(str(out_root / "per_band"))
		run_summary.coalesce(1).write.mode("overwrite").parquet(str(out_root / "run_summary"))

		print("Wrote telemetry aggregates:")
		print(f"- {out_root / 'per_weapon'}")
		print(f"- {out_root / 'per_band'}")
		print(f"- {out_root / 'run_summary'}")

		print("\nTop weapons by kills:")
		per_weapon.orderBy(F.desc("kills")).show(15, truncate=False)

	finally:
		spark.stop()


if __name__ == "__main__":
	main()
