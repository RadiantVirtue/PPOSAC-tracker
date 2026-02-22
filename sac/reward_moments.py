from sac.gradients import compute_reinterpreted_gradient


# split success transitions by reward sign (positive / neutral / negative)
def sub_partition_success(success_batch):
    rewards = success_batch["rewards"]

    result = {}
    for name, mask_fn in [
        ("positive", lambda r: r > 0),
        ("neutral", lambda r: r == 0),
        ("negative", lambda r: r < 0),
    ]:
        mask = mask_fn(rewards)
        if mask.any():
            result[name] = {
                "observations": success_batch["observations"][mask],
                "actions": success_batch["actions"][mask],
                "rewards": success_batch["rewards"][mask],
            }
        else:
            result[name] = None

    return result


def compute_reward_moment_gradients(
    actor, qf1, qf2, success_batch, alpha, device="cuda"
):
    # compute reinterpreted gradients for each reward subgroup
    subgroups = sub_partition_success(success_batch)
    gradients = {}

    for name, batch in subgroups.items():
        if batch is not None and len(batch["observations"]) > 0:
            gradients[name] = compute_reinterpreted_gradient(
                actor, qf1, qf2, batch, alpha, device
            )
        else:
            gradients[name] = None

    return gradients
