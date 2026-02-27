import os
import shutil
import torch
import torch.distributed as dist
from datetime import datetime
from utils.logger import setup_logger


def _broadcast_string(s: str, src: int = 0) -> str:
    """Broadcast a string from src rank to all ranks."""
    path_bytes = s.encode("utf-8") if dist.get_rank() == src else b""
    length = torch.tensor([len(path_bytes)], dtype=torch.long, device="cuda")
    dist.broadcast(length, src=src)
    if dist.get_rank() == src:
        path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device="cuda")
    else:
        path_tensor = torch.empty(length.item(), dtype=torch.uint8, device="cuda")
    dist.broadcast(path_tensor, src=src)
    return bytes(path_tensor.cpu().tolist()).decode("utf-8")


def setup_log_dir(cfg, args, log_dir_name=""):
    """Create timestamped log directory, logger, dump config, and copy source."""
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if log_dir_name:
        log_dir = os.path.join(output_dir, log_dir_name)
    else:
        now = datetime.now()
        log_dir = os.path.join(output_dir, now.strftime("%b_%Y"), now.strftime("%b%d_%H-%M-%S"))

    # Synchronize log_dir across DDP ranks so all use the same timestamp
    if dist.is_available() and dist.is_initialized():
        log_dir = _broadcast_string(log_dir, src=0)

    if args.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger("train", log_dir, if_train=True)
    if args.local_rank == 0:
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info(args)
        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, "r") as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
        with open(os.path.join(log_dir, "config.yml"), "w") as f:
            f.write(cfg.dump())

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source_dir = os.path.join(log_dir, "source")
        for folder in ("config", "configs", "datasets", "loss", "model", "processor", "solver", "utils"):
            src = os.path.join(project_root, folder)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(source_dir, folder))
        for script in ("train_pair.py", "train.py", "test.py"):
            src = os.path.join(project_root, script)
            if os.path.isfile(src):
                os.makedirs(source_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(source_dir, script))
        logger.info("Source code saved to {}".format(source_dir))

    if args.local_rank == 0:
        logger.info("Log directory: {}".format(log_dir))
    return log_dir, logger
