import torch
import torch.nn.functional as F

from shared.gradient_utils import OnlineGradientAggregator


# compute the gradient the current actor would produce on a batch
# returns L2-normalised mean gradient dict (layer name -> tensor)
def compute_reinterpreted_gradient(
    actor, qf1, qf2, batch, alpha, device="cuda"
):
    actor.train()
    aggregator = OnlineGradientAggregator(list(actor.named_parameters()))

    obs = batch["observations"]
    minibatch_size = 256

    for i in range(0, len(obs), minibatch_size):
        mb_obs = obs[i : i + minibatch_size].to(device)
        actor.zero_grad()

        logits = actor(mb_obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_pi = F.log_softmax(logits, dim=1)
        action_probs = dist.probs

        with torch.no_grad():
            q1 = qf1(mb_obs)
            q2 = qf2(mb_obs)
            min_q = torch.min(q1, q2)

        actor_loss = (action_probs * (alpha * log_pi - min_q)).mean()
        actor_loss.backward()

        aggregator.accumulate(list(actor.named_parameters()))
        actor.zero_grad()

    actor.eval()
    return aggregator.l2_normalized()
