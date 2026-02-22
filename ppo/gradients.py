import torch
import torch.distributions

from shared.gradient_utils import OnlineGradientAggregator


# compute GAE advantages for a single episode
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = (
            delta + gamma * lam * next_non_terminal * last_gae
        )

    return advantages


# online gradient aggregation of L^CLIP for a group of episodes
# returns L2-normalised mean gradient dict (layer name -> tensor)
def compute_group_gradient(agent, episodes, device="cuda"):
    agent.train()
    aggregator = OnlineGradientAggregator(list(agent.named_parameters()))

    for episode in episodes:
        agent.zero_grad()
        obs = episode.observations.to(device)
        actions = episode.actions.to(device)

        embedding = agent._embed(obs)
        logits = agent.actor(embedding)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        values = agent.critic(embedding).squeeze(-1)

        advantages = compute_gae(
            episode.rewards, values.detach(), episode.dones
        ).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = -(advantages * log_probs).mean()
        pg_loss.backward()

        aggregator.accumulate(list(agent.named_parameters()))
        agent.zero_grad()

    agent.eval()
    return aggregator.l2_normalized()


def compute_group_gradient_with_coherence(
    agent, episodes, batch_size=50, device="cuda"
):
    # like compute_group_gradient but also yields minibatch-level gradient vectors for coherence
    # returns (l2_normalised_mean, list_of_minibatch_gradient_dicts)
    agent.train()
    overall_agg = OnlineGradientAggregator(list(agent.named_parameters()))
    minibatch_grads = []

    for i in range(0, len(episodes), batch_size):
        batch = episodes[i : i + batch_size]
        batch_agg = OnlineGradientAggregator(list(agent.named_parameters()))

        for episode in batch:
            agent.zero_grad()
            obs = episode.observations.to(device)
            actions = episode.actions.to(device)

            embedding = agent._embed(obs)
            logits = agent.actor(embedding)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            values = agent.critic(embedding).squeeze(-1)

            advantages = compute_gae(
                episode.rewards, values.detach(), episode.dones
            ).to(device)
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            pg_loss = -(advantages * log_probs).mean()
            pg_loss.backward()

            batch_agg.accumulate(list(agent.named_parameters()))
            overall_agg.accumulate(list(agent.named_parameters()))
            agent.zero_grad()

        minibatch_grads.append(batch_agg.l2_normalized())

    agent.eval()
    return overall_agg.l2_normalized(), minibatch_grads
