from agents import Agent,Runner,RunConfig,AsyncOpenAI,OpenAIChatCompletionsModel,function_tool
from dotenv import load_dotenv,find_dotenv
import os

load_dotenv(find_dotenv(),override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_base_url = os.getenv("BASE_URL")
gemini_ai_model = os.getenv("AI_MODEL")


external_client = AsyncOpenAI(api_key=gemini_api_key,base_url=gemini_base_url)

Model = OpenAIChatCompletionsModel(openai_client=external_client,model=gemini_ai_model)



@function_tool
def addition(a:int,b:int)-> str|int :
    """ add two numbers and give simple string output
        """
    return f" after addition result is {a+b} "


@function_tool
def multiplication(a:int,b:int)->str|int :
    """ Multiply two numbers and give simple string output
        """
    return f" after multiplication result is {a*b} "


@function_tool
def division(a:int,b:int)-> str|int :
    """ Divide two numbers and give simple string output
        """
    return f" after division result is {a/b} "







math_agent = Agent(
    name="Math Agent",
    instructions="As a math agent, you only respond to user prompts by utilizing the appropriate tool.  If there is no tool for a question, just say 'Sorry'.",
    tools=[addition,multiplication,division]
)

config = RunConfig(model=Model,model_provider=external_client,tracing_disabled=True)

prompt = input("Enter Your Question ")
result = Runner.run_sync(math_agent,prompt,run_config=config)

print(result.final_output)