"""Script maestro del analisis de sensibilidad.

Uso recomendado desde la raiz del repo:

    python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --dry-run
    python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py

La ejecucion completa reentrena modelos temporales de XGBoost para los
escenarios 1, 2 y 4. Puede tardar mas de 10 minutos segun el equipo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.sensitivity import (  # noqa: E402
    escenario_1_umbrales_plos,
    escenario_2_ablation_features,
    escenario_3_punto_operacion,
    escenario_4_hiperparametros,
    smoke_test,
)
from ml_operacional_entrega3.sensitivity.common import (  # noqa: E402
    RESULTS_DIR,
    SensitivityOutput,
    ensure_results_dir,
    existing_required_csvs,
    expected_output_files,
    load_holdout_predictions_with_prob,
    read_baseline_mae,
    require_all_result_csvs,
    result_files_are_valid,
    validate_result_contracts,
)
from ml_operacional_entrega3.sensitivity.reporting import generate_consolidated_report  # noqa: E402
from ml_operacional_entrega3.utils.pipeline_operacional import (  # noqa: E402
    DATA_SPLITS_DIR,
    load_segment_split,
)


SCENARIOS = {
    "1": ("Escenario 1 - Umbrales PLOS", escenario_1_umbrales_plos.run),
    "2": ("Escenario 2 - Ablation study", escenario_2_ablation_features.run),
    "3": ("Escenario 3 - Punto de operacion", escenario_3_punto_operacion.run),
    "4": ("Escenario 4 - Hiperparametros vecinos", escenario_4_hiperparametros.run),
}

SCENARIO_OUTPUTS = {
    "1": ["escenario_1_resultados.csv", "escenario_1_resultados_por_tramo.csv"],
    "2": ["escenario_2_resultados.csv"],
    "3": ["escenario_3_puntos_operacion.csv", "escenario_3_curva_pr.csv"],
    "4": ["escenario_4_resultados.csv"],
}


def log(message: str = "") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}" if message else "", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el analisis de sensibilidad XGB.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Ejecuta solo un escenario. Se puede repetir. Por defecto ejecuta 1,2,3,4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida rutas y muestra el plan sin entrenar modelos ni escribir resultados.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="No ejecuta escenarios; genera solo el reporte si ya existen todos los CSV.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Ejecuta escenarios pero no genera el reporte consolidado.",
    )
    parser.add_argument(
        "--validate-results",
        action="store_true",
        help="Valida que los CSVs generados existan, no esten vacios y tengan las columnas requeridas.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Ejecuta una prueba de humo liviana sin escribir outputs finales.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Salta escenarios cuyos CSVs ya existen y cumplen el contrato de columnas.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Muestra estado de outputs por escenario sin ejecutar entrenamientos.",
    )
    return parser.parse_args()


def _print_expected_outputs() -> None:
    log("")
    log("Salidas esperadas:")
    for idx, path in enumerate(expected_output_files(), 1):
        log(f"  {idx}. {path}")
    log(f"Total de salidas esperadas: {len(expected_output_files())}")


def dry_run() -> None:
    log("=== Dry run del analisis de sensibilidad ===")
    ensure_results_dir()
    log(f"Directorio results: {RESULTS_DIR}")
    log(f"Splits operacionales: {DATA_SPLITS_DIR}")

    for segment in ["urgente", "programado"]:
        train = load_segment_split(segment, "train")
        holdout = load_segment_split(segment, "holdout")
        log(
            f"  {segment}: train={train.shape}, holdout={holdout.shape}, "
            f"PLOS_train={(train['los_dias'] >= 14).sum()}, PLOS_holdout={(holdout['los_dias'] >= 14).sum()}"
        )
        for threshold in [7, 14, 21, 27]:
            log(
                f"    umbral {threshold}: "
                f"train_pos={(train['los_dias'] >= threshold).sum()}, "
                f"holdout_pos={(holdout['los_dias'] >= threshold).sum()}"
            )

    baseline_mae = read_baseline_mae()
    log(f"MAE base XGB detectado: {baseline_mae:.4f}")

    holdout_predictions = load_holdout_predictions_with_prob()
    log(
        "Predicciones holdout XGB disponibles: "
        f"{holdout_predictions.shape[0]} filas, {holdout_predictions.shape[1]} columnas"
    )
    existing = existing_required_csvs()
    log(f"CSVs de sensibilidad ya existentes: {len(existing)}")
    for path in existing:
        log(f"  - {path}")
    _print_expected_outputs()
    log("")
    log("Dry run completado. No se entrenaron modelos ni se generaron reportes.")


def run_selected_scenarios(selected: list[str], *, skip_existing: bool = False) -> list[SensitivityOutput]:
    outputs: list[SensitivityOutput] = []
    for scenario_id in selected:
        name, runner = SCENARIOS[scenario_id]
        if skip_existing:
            valid, issues = result_files_are_valid(SCENARIO_OUTPUTS[scenario_id])
            if valid:
                log("")
                log(f"Saltando {name}: outputs existentes y validos.")
                continue
            log("")
            log(f"{name} se ejecutara; outputs incompletos: {'; '.join(issues)}")
        log("")
        log("############################################################")
        log(f"Ejecutando {name}")
        log("############################################################")
        start = time.time()
        scenario_outputs = runner()
        outputs.extend(scenario_outputs)
        elapsed = time.time() - start
        rows = sum(output.rows for output in scenario_outputs)
        log(f"{name} completado en {elapsed / 60:.2f} min | salidas={len(scenario_outputs)} | filas={rows}")
    return outputs


def write_execution_manifest(
    *,
    args: argparse.Namespace,
    selected: list[str],
    outputs: list[SensitivityOutput],
    elapsed_seconds: float,
) -> Path:
    manifest_path = RESULTS_DIR / "execution_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_scenarios": selected,
        "options": {
            "dry_run": bool(args.dry_run),
            "report_only": bool(args.report_only),
            "skip_report": bool(args.skip_report),
            "validate_results": bool(args.validate_results),
            "smoke_test": bool(args.smoke_test),
            "skip_existing": bool(args.skip_existing),
        },
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "elapsed_minutes": round(float(elapsed_seconds) / 60.0, 3),
        "outputs_generated_this_run": len(outputs),
        "rows_or_lines_generated_this_run": int(sum(output.rows for output in outputs)),
        "outputs": [
            {
                "path": str(output.path),
                "name": output.path.name,
                "rows_or_lines": int(output.rows),
            }
            for output in outputs
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Manifest de ejecucion guardado: {manifest_path}")
    return manifest_path


def print_status() -> None:
    log("=== Estado de outputs de sensibilidad ===")
    complete_count = 0
    for scenario_id in sorted(SCENARIOS.keys()):
        name, _ = SCENARIOS[scenario_id]
        filenames = SCENARIO_OUTPUTS[scenario_id]
        valid, issues = result_files_are_valid(filenames)
        if valid:
            complete_count += 1
            log(f"[OK] {name}")
        else:
            log(f"[PENDIENTE] {name}")
            for issue in issues:
                log(f"  - {issue}")

    report_path = RESULTS_DIR / "reporte_sensibilidad_consolidado.md"
    report_ok = report_path.exists() and report_path.stat().st_size > 0
    log("")
    log(f"Escenarios completos: {complete_count}/{len(SCENARIOS)}")
    log(f"Informe consolidado: {'OK' if report_ok else 'PENDIENTE'}")
    if complete_count == len(SCENARIOS) and not report_ok:
        log("Siguiente paso: python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --report-only")
    elif complete_count < len(SCENARIOS):
        log("Siguiente paso: python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --skip-report --skip-existing")


def main() -> None:
    args = parse_args()
    ensure_results_dir()

    if args.dry_run:
        dry_run()
        return

    if args.status:
        print_status()
        return

    if args.smoke_test:
        smoke_test.run()
        return

    if args.validate_results:
        validation = validate_result_contracts(require_report=True)
        if (validation["estado"] != "ok").any():
            raise SystemExit("La validacion encontro outputs faltantes o invalidos.")
        return

    if args.report_only:
        log("=== Generando solo reporte consolidado ===")
        selected = []
        start = time.time()
        require_all_result_csvs()
        output = generate_consolidated_report()
        validation = validate_result_contracts(require_report=True)
        if (validation["estado"] != "ok").any():
            raise SystemExit("El reporte fue generado, pero la validacion encontro problemas.")
        write_execution_manifest(
            args=args,
            selected=selected,
            outputs=[output],
            elapsed_seconds=time.time() - start,
        )
        log(f"Reporte generado: {output.path}")
        return

    selected = args.scenario if args.scenario else sorted(SCENARIOS.keys())
    log("=== Analisis de sensibilidad XGB ===")
    log(f"Escenarios a ejecutar: {', '.join(selected)}")
    log("Nota: los modelos entrenados aqui son temporales y no sobrescriben modelos_guardados/.")
    _print_expected_outputs()

    start = time.time()
    outputs = run_selected_scenarios(selected, skip_existing=args.skip_existing)

    if not args.skip_report and set(selected) == set(SCENARIOS.keys()):
        log("")
        log("=== Generando reporte consolidado ===")
        outputs.append(generate_consolidated_report())
        validation = validate_result_contracts(require_report=True)
        if (validation["estado"] != "ok").any():
            raise SystemExit("La validacion final encontro outputs faltantes o invalidos.")
    elif not args.skip_report:
        log("")
        log("Reporte consolidado omitido porque no se ejecutaron los 4 escenarios en esta corrida.")
        log("Cuando existan todos los CSV, ejecuta: python3 ml_operacional_entrega3/sensitivity/run_sensitivity.py --report-only")
        validate_result_contracts(require_report=False)
    elif set(selected) == set(SCENARIOS.keys()):
        log("")
        log("=== Validando CSVs generados sin informe ===")
        validation = validate_result_contracts(require_report=False)
        csv_validation = validation[validation["archivo"].astype(str).str.endswith(".csv")]
        if (csv_validation["estado"] != "ok").any():
            raise SystemExit("La validacion encontro CSVs faltantes o invalidos.")
    else:
        log("")
        log("Validacion global omitida porque se ejecuto un subconjunto de escenarios con --skip-report.")

    elapsed = time.time() - start
    total_rows = sum(output.rows for output in outputs)
    log("")
    log("=== Resumen final de ejecucion ===")
    log(f"Tiempo total: {elapsed / 60:.2f} min")
    log(f"Salidas generadas en esta corrida: {len(outputs)}")
    log(f"Filas/lineas totales generadas: {total_rows}")
    for output in outputs:
        log(f"  - {output.path} ({output.rows})")
    write_execution_manifest(
        args=args,
        selected=selected,
        outputs=outputs,
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    main()
