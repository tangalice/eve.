import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

user_prompt = "Always"

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
                    You are a factual and informational assistant. When given the name of a feminine product such as a pad or tampon brand, provide a 1-sentence description of the following: 
                    1. Beneficial ingredients included in the product if there are any. If not, skip this line. Description title: "Beneficial Ingredients"
                    2. Harmful ingredients included in the product. Description title: "Harmful Ingredients"
                    3. Potential side effects of using the product. Description title: "Potential Side Effects"
                    4. Suggested safer alternatives to the product. Description title: "Safer Alternatives"
                    5. A ranking on a scale from 1-5 of how safe it is to use, with 1 highly unsafe and 5 being mostly safe. Description title: "Ranking"
                    Do not use overly complex medical jargon, but still communicate the key information. 
                    Start with the name of the product, then a new line. 
                    Then each description in a bulleted list, separated by a new line. Use the description title to start each bullet.
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