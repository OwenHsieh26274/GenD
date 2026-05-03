from .. import config as C
from ..config import Config
from ..utils import files

experiments = {
    "sdfvd2-DINOv3B-LN+L2+UA": [
        Config(
            run_dir="runs/sdfvd2",
            wandb=True,
            wandb_tags=["sdfvd2", "dinov3", "gend"],
            devices="auto",
            num_workers=8,
            backbone=C.Backbone.DINOv3_ViT_B,
            freeze_feature_extractor=True,
            unfreeze_layers=["norm1", "norm2", "norm"],
            head=C.Head.NLinear,
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.5),
            trn_files=files.SDFVD2_0.train,
            val_files=files.SDFVD2_0.val,
            tst_files={"SDFVD2.0": files.SDFVD2_0.test},
            lr=3e-4,
            min_lr=1e-5,
            lr_scheduler="cyclic",
            num_epochs_in_cycle=10,
            weight_decay=0.0,
            max_epochs=30,
            warmup_epochs=1,
            batch_size=128,
            mini_batch_size=16,
            precision="bf16-mixed",
            throw_exception_if_run_exists=True,
        )
    ],
}
