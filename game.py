import gymnasium as gym

from imprisoned_env import ImprisonedEnv


def play_game():
    """Runs the text-based game loop for 'Imprisoned'."""
    env = ImprisonedEnv()
    observation = env.reset()
    env.render()

    done = False
    while not done:
        actions = env.get_available_actions()

        if not actions:
            print("\nNo actions available. Game over!")
            break

        print("\nAvailable actions:")
        for i, action in enumerate(actions, start=1):
            action_details = env.states[env.current_state]["actions"][action]
            print(f"{i}. {action}: {action_details.get('description', 'No description')}")

        try:
            choice = int(input("\nChoose an action (number): ")) - 1
            if 0 <= choice < len(actions):
                observation, reward, done, _ = env.step(choice)
                print("\n" + "=" * 50)  # Add visual separator
                env.render()
            else:
                print("Invalid choice. Please pick a valid number.")
        except ValueError:
            print("Invalid input. Enter a number corresponding to an action.")

    print("\n=== GAME OVER ===")
    if observation == "escape_success":
        print("🎉 You have successfully escaped! Congratulations!")
    elif reward == -1:
        state_desc = env.get_state_description()
        print(f"💀 Game Over: {state_desc}")
    else:
        print("💀 You failed to escape. Better luck next time!")


if __name__ == "__main__":
    print("\n=== Welcome to *Imprisoned* ===")
    play_game()
