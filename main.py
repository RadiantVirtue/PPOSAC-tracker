# Available environments:
# "MiniGrid-Empty-5x5-v0"
# "MiniGrid-Empty-8x8-v0"
# "MiniGrid-Empty-16x16-v0"
# "MiniGrid-DoorKey-5x5-v0"
# "MiniGrid-DoorKey-8x8-v0"
# "MiniGrid-FourRooms-v0"
# "MiniGrid-LavaGapS5-v0"
# "MiniGrid-SimpleCrossingS9N1-v0"
ENV_ID = "MiniGrid-Empty-5x5-v0"

# Available algorithms: "ppo" or "sac"
ALGORITHM = "ppo"

TOTAL_TIMESTEPS = 500000
SEED = 1
CAPTURE_VIDEO = False
TRACK_WANDB = False

if __name__ == "__main__":
    print(f"Running {ALGORITHM.upper()} on {ENV_ID}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print("-" * 50)

    if ALGORITHM == "ppo":
        from ppo_minigrid import Args, main_ppo
        args = Args(
            env_id=ENV_ID,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=SEED,
            capture_video=CAPTURE_VIDEO,
            track=TRACK_WANDB,
        )
        main_ppo(args)

    elif ALGORITHM == "sac":
        from sac_minigrid import Args, main_sac
        args = Args(
            env_id=ENV_ID,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=SEED,
            capture_video=CAPTURE_VIDEO,
            track=TRACK_WANDB,
        )
        main_sac(args)

    else:
        print(f"Unknown algorithm: {ALGORITHM}. Choose 'ppo' or 'sac'.")
