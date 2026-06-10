print("<b>КЛАСТЕРИЗАЦИЯ ФОТОГРАФИЙ ПО СЮЖЕТАМ</b>")
print("<i>Инициализация...</i><br>")

import logging
import sys
from pathlib import Path
import argparse

IS_MANAGED_RUN = False
try:
    CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

    if str(CURRENT_SCRIPT_DIR) not in sys.path: 
        sys.path.insert(0, str(CURRENT_SCRIPT_DIR))

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pysm_lib import pysm_context
    from pysm_lib.pysm_context import ConfigResolver
    from pysm_lib.pysm_report_api import ResourceNode, StandardTreeBuilder, DashboardBuilder
    from pysm_lib.pysm_icons import icons as pysm_icons    
    
    IS_MANAGED_RUN = True

except ImportError as e:
    print(f"Критическая ошибка импорта внутренних модулей: {e}", file=sys.stderr)
    pysm_icons = None
    sys.exit(1)



from cluster_locations2.application.pipeline import PipelineError, run_pipeline
from cluster_locations2.config.config import ConfigManager




def get_config() -> argparse.Namespace:
    

    parser = argparse.ArgumentParser(description="Кластеризация фотографий по локациям")

    default_config_path = Path(__file__).parent / "config.toml"
    p = "a_cl_"

    parser.add_argument(f"--{p}config_file", type=str, default=str(default_config_path))
    parser.add_argument(f"--{p}data_dir", type=str, required=True)

    parser.add_argument("--mode", type=str, default="clustering", choices=["clustering", "classification"])
    parser.add_argument(
        "--model_backend",
        type=str,
        default=None,
        choices=["clip", "siglip2_onnx"],
    )
    parser.add_argument(
        "--spatial_strategy",
        type=str,
        default=None,
        choices=["flatten_axis1_norm", "flatten", "grid_9x9", "grid_6x6", "grid_3x3", "mean_std", "pooler"],
    )
    parser.add_argument("--match_threshold", type=float, default=0.11)
    parser.add_argument("--use_originals", action="store_true")

    parser.add_argument(f"--{p}location_prompts", type=str, nargs='*', default=[])
    parser.add_argument(f"--{p}cluster_eps", type=float, default=0.18)
    parser.add_argument(f"--{p}mask_suffix", type=str, default=None)

    parser.add_argument("--all_threads", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_mode", type=str, default=None, choices=["use", "refresh", "off"])

    return ConfigResolver(parser).resolve_all()


def main():
    #logging.basicConfig(level=logging.INFO, format="%(message)s")


    log_level = "INFO"
    if IS_MANAGED_RUN and pysm_context:
        log_level = pysm_context.get("sys_log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s", stream=sys.stdout)

    cfg = get_config()

    config_path = Path(getattr(cfg, "a_cl_config_file"))
    data_dir = Path(getattr(cfg, "a_cl_data_dir"))

    config_manager = ConfigManager(config_path)
    config = config_manager.config

    # --- CLI overrides ---

    if cfg.model_backend:
        config.model.backend = cfg.model_backend

    if cfg.spatial_strategy:
        config.siglip2_onnx.spatial_strategy = cfg.spatial_strategy

    config_manager.apply_backend_defaults()

    if cfg.a_cl_cluster_eps is not None:
        config.clustering.eps = cfg.a_cl_cluster_eps

    if cfg.a_cl_mask_suffix:
        config.model_params.mask_suffix = cfg.a_cl_mask_suffix

    if cfg.a_cl_location_prompts:
        config.classification.prompts = cfg.a_cl_location_prompts

    config.classification.match_threshold = cfg.match_threshold

    if cfg.cache_mode:
        config.cache.mode = cfg.cache_mode

    # threads
    workers = cfg.all_threads if cfg.all_threads > 0 else 4

    # логика mask/original (ВАЖНО)
    input_is_mask = not cfg.use_originals

    try:
        run_pipeline(
            data_dir=data_dir,
            config=config,
            mode=cfg.mode,
            input_is_mask=input_is_mask,
            workers=workers,
            batch_size=cfg.batch_size,
            cache_mode=config.cache.mode,
        )
        photo_session = pysm_context.get("ws_photo_session", "SCHOOL")        
        pysm_context.set_structured(f"var_claster_run.{photo_session}.location", "yes")

    except PipelineError as e:
        logging.error(f"<br><b>Pipeline error:</b> {e}<br>")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"<br><b>Unexpected error:</b> {e}<br>")
        sys.exit(1)


    tv_builder = StandardTreeBuilder(icon_size=28)
    root_node1 = ResourceNode("config.toml", Path(cfg.a_cl_config_file), "txt", "Конфиг")
    root_node = ResourceNode("Data", Path(cfg.a_cl_data_dir), "folder", "Данные")
    tv_builder.add_section("Ресурсы", [root_node1, root_node])
    pysm_context.log_html(tv_builder.get_html())    


if __name__ == "__main__":
    main()
