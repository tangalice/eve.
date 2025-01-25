import os
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, render_template, request

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

app = Flask(__name__)

global output
sections = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global output
    global sections
    if request.method == 'POST':
        search_query = request.form['search']
        # Here, you can do whatever you want with the search_query
        chat_completion = client.chat.completions.create(
            messages=[
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
            {
                "role": "user",
                "content": search_query,
            }
            ],

            model="llama-3.3-70b-versatile",

            temperature=0.5,

            max_completion_tokens=1024,

            top_p=1,

            stop=None,

            stream=False,
        )
        output = chat_completion.choices[0].message.content
        sections = [section.strip() for section in output.split('*') if section.strip()]

        print(output)
        # You can also save it to a file if you prefer
        with open("output.txt", "w") as f:
            f.write(output)
    
    # Render the HTML page and pass the output to it
    return render_template('index.html', sections=sections)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
