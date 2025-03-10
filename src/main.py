from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage




giga = GigaChat(
    credentials=giga_config.GIGACHAT_TOKEN,
    model=GigaModels.GIGACHAT_PRO,
    scope=GigaScopes.CORP,
    verify_ssl_certs=False
)

messages_to_giga = [
    SystemMessage(
        content='Ты чикагский гангстер. Отвечай цитатамии из фильмов'
    )
]

while True:
    user_input = input("Пользователь: ")
    if not user_input:
        break
    messages_to_giga.append(HumanMessage(content=user_input))
    res = giga.invoke(messages_to_giga)
    messages_to_giga.append(res)
    print("Gigachat: ", res.content)