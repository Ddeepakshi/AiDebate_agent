from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from dotenv import load_dotenv
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
import os
import asyncio

load_dotenv()  # Load variables from .env file
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


async def main():
    model=AnthropicChatCompletionClient(
        model="claude-3-5-sonnet-20241022",
        
        api_key = os.getenv('API_KEY'),
    )

    topic = "Should AI be regulated by the government?"
    host= AssistantAgent(
        name="Host",    
        model_client=model,  # Enable streaming tokens from the model client.
        system_message=(
            f'You are the host of a debate  between john , a supporter agent and jack a critic agent on the topic: ' + topic + '. You will moderate the debate .At the beginning of each round , announce the round numberand at the third round declre that it will be the last round of the debate. After the last round, summarize the debate and declare the winner based on the arguments presented.You will also provide a brief summary of the arguments presented by each side at the end of the debate.'
        ),  
    )    
    supporter = AssistantAgent(
        name="John",
        model_client=model,
        system_message=(
            f'You are John, a supporter agent in a debate for the topic {topic}. You will be debating against Jack, a critic agent.'
        ),
          # Enable streaming tokens from the model client.
    )
    critic = AssistantAgent(
        name="Jack",
        model_client=model,
        system_message=(
            f'You are a Jack , critic agent in a debate for the topic {topic}. you will be debating agent against john , a supporter agent. '
        ),
  # Enable streaming tokens from the model client.
    )

    team=RoundRobinGroupChat(
        participants=[host,supporter, critic],
        max_turns=10
    )

    async for message in team.run_stream(task="Start the debate on the topic: " + topic,):
        print('--' * 20)
        if isinstance(message, TaskResult):
            print(f"Stopping reason :{message.stop_reason}")
        else:
            if hasattr(message, 'source'):
                print(f"{message.source}: {message.content}")
            else:
                print(f"Message: {message.content}")

    # res= await team.run(task="Start the debate on the topic: " + topic,)
    # for message in res.messages:
    #     print('--'*20)
    #     print(f"{message.sender.name}: {message.content}")
if __name__=="__main__":
    asyncio.run(main())