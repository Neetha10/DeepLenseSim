from agent_module.agent import simulation_agent


def chat():
    print("=" * 50)
    print("DeepLenseSim Agent ready!")
    print("=" * 50)
    print("You can:")
    print("  - Request simulations: 'Generate 5 CDM images using Model_I'")
    print("  - List models: 'What models are available?'")
    print("  - Compare models: 'Compare CDM across all models'")
    print("  - Type 'quit' to exit")
    print("=" * 50 + "\n")

    message_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        result = simulation_agent.run_sync(
            user_input,
            message_history=message_history
        )

        response = result.output
        print(f"\nAgent: {response}\n")

        # use pydantic-ai's own message history
        message_history = result.all_messages()