from . import (
    cddd,
    examples,
    sdfvd2,
    third_party,
    wacv_rebuttal,
    wacv_rebuttal_aug_robustness,
    wacv_rebuttal_paired_unpaired,
)

experiments = {
    **cddd.experiments,
    **examples.experiments,
    **sdfvd2.experiments,
    **third_party.experiments,
    **wacv_rebuttal.experiments,
    **wacv_rebuttal_paired_unpaired.experiments,
    **wacv_rebuttal_aug_robustness.experiments,
}
