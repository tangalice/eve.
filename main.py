import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

user_prompt = "Kotex"

chat_completion = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "assistant",
            "content": """
                    You are a factual and informational assistant. When given the name of a feminine product such as a pad or tampon brand, provide a detailed description of the following: 
                    1. Harmful ingredients included in the product.
                    2. Potential side effects of using the product.
                    3. Suggested cleaner alternatives to the product.
                    4. A ranking on a scale from 1-5 of how safe it is to use, with 1 being mostly safe and 5 being highly unsafe.
                """
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": user_prompt,
        }
    ],

    # The language model which will generate the completion.
    model="llama-3.3-70b-versatile",

    #
    # Optional parameters
    #

    # Controls randomness: lowering results in less random completions.
    # As the temperature approaches zero, the model will become deterministic
    # and repetitive.
    temperature=0.5,

    # The maximum number of tokens to generate. Requests can use up to
    # 32,768 tokens shared between prompt and completion.
    max_completion_tokens=1024,

    # Controls diversity via nucleus sampling: 0.5 means half of all
    # likelihood-weighted options are considered.
    top_p=1,

    # A stop sequence is a predefined or user-specified text string that
    # signals an AI to stop generating content, ensuring its responses
    # remain focused and concise. Examples include punctuation marks and
    # markers like "[end]".
    stop=None,

    # If set, partial message deltas will be sent.
    stream=False,
)

# Print the completion returned by the LLM.
print(chat_completion.choices[0].message.content)