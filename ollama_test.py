import ollama

response = ollama.chat(
    model='llama3.1',
    messages=[
        {
            'role': 'user',
            'content': 'Explique o que é inteligência artificial em uma frase.'
        }
    ]
)

print(response['message']['content'])