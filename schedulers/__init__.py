import logging

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from schedulers.schedulers import WarmUpLR, ConstantLR, PolynomialLR

key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "step": StepLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
    "reduce_on_plateau": ReduceLROnPlateau,
    "cycle_lr": CyclicLR,
}

def select_scheduler(optimizer, params, train_or_search):
    # train_or_search is either "train" or "search" (str)
    #print(params)
    if params is None:
        print("Using constant LR Scheduling")
        return ConstantLR(optimizer)

    scheduler_dict = params.copy()
    s_type = scheduler_dict["name"]
    scheduler_dict.pop("name")

    print("Using {} scheduler with {} params".format(s_type, scheduler_dict))

    # We need to have a separate schedule for searching and training as we have different number of epochs
    if s_type == "cosine_annealing":

        if train_or_search == "train":
            scheduler_dict["T_max"] = scheduler_dict["T_max_train"]
            scheduler_dict.pop("T_max_search")
            scheduler_dict.pop("T_max_train")
        elif train_or_search == "search":
            scheduler_dict["T_max"] = scheduler_dict["T_max_search"]
            scheduler_dict.pop("T_max_search")
            scheduler_dict.pop("T_max_train")

    warmup_dict = {}
    if "warmup_iters" in scheduler_dict:
        # This can be done in a more pythonic way...
        warmup_dict["warmup_iters"] = scheduler_dict.get("warmup_iters", 100)
        warmup_dict["mode"] = scheduler_dict.get("warmup_mode", "linear")
        warmup_dict["gamma"] = scheduler_dict.get("warmup_factor", 0.2)

        print(
            "Using Warmup with {} iters {} gamma and {} mode".format(
                warmup_dict["warmup_iters"], warmup_dict["gamma"], warmup_dict["mode"]
            )
        )

        scheduler_dict.pop("warmup_iters", None)
        scheduler_dict.pop("warmup_mode", None)
        scheduler_dict.pop("warmup_factor", None)

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)


    return key2scheduler[s_type](optimizer, **scheduler_dict)
