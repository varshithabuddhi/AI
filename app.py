import os
from flask import Flask, render_template, request
from markupsafe import Markup
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

app = Flask(__name__)

key = os.environ.get("GROQ_API_KEY")#key
chat_groq = ChatGroq(api_key=key,model='llama-3.3-70b-versatile')

memory = ConversationBufferWindowMemory(k=5)
conversation = ConversationChain(
    llm=chat_groq,
    memory=memory
)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    assignment = ""
    if request.method == 'POST':
        topic = request.form.get('topic')
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Generate a short 3-question assignment on the topic: {topic}"
        )
        raw_assignment = conversation.run(prompt.format(topic=topic))
        # Convert newlines and formatting to HTML
        assignment = Markup(raw_assignment.replace('\n', '<br>'))
    return render_template("generate.html", assignment=assignment)


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    result = {}
    if request.method == 'POST':
        answer = request.form.get('answer')
        eval_prompt = PromptTemplate(
            input_variables=["answer"],
            template="""
            You are an expert and a strict teacher and strict AI detector.
            Evaluate this student's answer: "{answer}".
            1. Provide constructive feedback.
            2. Give a score out of 5.
            3. Predict whether this answer is AI-generated (Yes or No).
            Be very strict while verifying answer is AI-generated
            Return the output in JSON format like:
            {{
                "feedback": "...",
                "score": 4,
                "ai_generated": "Yes/No"
            }}
            """
        )
        response = conversation.run(eval_prompt.format(answer=answer))


        import json
        try:
            result = json.loads(response)
        except:
            result = {"feedback": response, "score": "N/A", "ai_generated": "N/A"}

    return render_template("evaluate.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)