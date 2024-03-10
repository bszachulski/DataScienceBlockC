from flask import Flask, render_template, request
from openai.embeddings_utils import cosine_similarity, get_embedding
import pandas as pd
import openai
import os

openai.api_key = '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'
os.environ["OPENAI_API_KEY"] = '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'

app = Flask(__name__)

# Sample dataframe with context and embeddings
# Replace this with your actual data
df = pd.DataFrame({
    'context': ['This is the first context.', 'This is the second context.', 'This is the third context.'],
    'embedding': [get_embedding('This is the first context.'), get_embedding('This is the second context.'), get_embedding('This is the third context.')]
})


def get_answer(question):
    question_vector = get_embedding(question, engine='text-embedding-ada-002')
    df["similarities"] = df['embedding'].apply(
        lambda x: cosine_similarity(x, question_vector))
    df = df.sort_values("similarities", ascending=False).head(2)

    context = df['context'].tolist()
    text = "\n".join(context)

    prompt = f"""Answer the following question using only the context below. 

    Context:
    {text}

    Q: {question}
    A:"""

    response = openai.Completion.create(
        prompt=prompt,
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="gpt-3.5-turbo-instruct"
    )

    return response["choices"][0]["text"].strip(" \n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']
    answer_text = get_answer(question)
    return render_template('answer.html', question=question, answer=answer_text)


if __name__ == "__main__":
    app.run(debug=True)
