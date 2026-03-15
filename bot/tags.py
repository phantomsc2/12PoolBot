from collections.abc import Awaitable, Callable

ChatFunction = Callable[[str], Awaitable]


class Tags:
    chat_function: ChatFunction

    def __init__(self, chat_function: ChatFunction) -> None:
        self.chat_function = chat_function
