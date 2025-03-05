from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import List, Dict
import httpx
from dotenv import load_dotenv
from browser_use import Browser, BrowserConfig, Agent as BrowserAgent, Controller
from langchain_openai import ChatOpenAI
import asyncio
# import logfire

# logfire.configure()

load_dotenv()


class LinkStatus(BaseModel):
    link: str
    status: bool


class LinkStatusList(BaseModel):
    links: List[LinkStatus]


link_checker = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant that checks the validity of links."
    "You will be given a plain text in which links may be present."
    "You need to extract the links and check their validity."
    "Return the status of the links in as a dictionary with links as keys and boolean values indicating whether the link is valid or not."
    "Use the check_link_list tool to check the validity of links.",
    result_type=LinkStatusList,
    retries=6,
)

corrector = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=(
        "You're an assistant that carefully edits text containing links. "
        "You will be given a text with links and a list of links with their status (valid or invalid). "
        "Your task is to remove ONLY the invalid links while preserving the original text as much as possible. "
        "Follow these guidelines: "
        "1. If an invalid link is part of a sentence, remove only the link itself, keeping the surrounding text intact. "
        "2. If an invalid link is presented as a standalone item (like in a list), remove only that item. "
        "3. Make the minimal changes necessary to maintain text coherence after removing invalid links. "
        "4. Do not modify, improve, or change any other parts of the text. "
        "5. Keep all valid links exactly as they appear in the original text."
        "Use the get_invalid_links tool to get the invalid links."
        "Respond only with the corrected text, nothing else."
    ),
    deps_type=LinkStatusList,
    result_type=str,
    retries=6,
)


class AvailabilityController(BaseModel):
    site_availability_status: bool
    text_from_site: str

@link_checker.tool_plain
async def check_link_list(list_of_links: List[str]) -> LinkStatusList:
    """
    Проверяет доступность ссылок с помощью сервиса browser-use.
    Для каждой ссылки агент открывает вкладку в браузере и, если страница загрузилась (возвращается непустой ответ),
    считается, что сайт доступен.
    """
    statuses = []
    browser_config = BrowserConfig(headless=True, disable_security=True)
    browser = Browser(config=browser_config)
    controller = Controller(output_model=AvailabilityController)

    llm = ChatOpenAI(model="gpt-4o")

    for link in list_of_links:
        try:
            initial_actions = [{'open_tab': {'url': link}}]
            task = f"Check this website {link} and verify that it is availability. Return any tiny part of text from the page. If you see CAPTCHA - then return site_availability_status 'false' and stop the work."
            agent = BrowserAgent(
                task=task,
                llm=llm,
                initial_actions=initial_actions,
                controller=controller,
                browser=browser
            )
            history = await agent.run()
            result = history.final_result()
            if result:
                parsed_result: AvailabilityController = AvailabilityController.model_validate_json(result)
                is_available = parsed_result.site_availability_status
            else:
                is_available = False
            statuses.append(LinkStatus(link=link, status=is_available))
        except Exception as e:
            print(f'Exception {e} ', link)
            statuses.append(LinkStatus(link=link, status=False))
    await browser.close()
    return LinkStatusList(links=statuses)

@corrector.tool
async def get_invalid_links(ctx: RunContext[LinkStatusList]) -> List[str]:
    if not ctx.deps or not hasattr(ctx.deps, "links"):
        return []
    return [link_status.link for link_status in ctx.deps.links if not link_status.status]


async def main():
    text = "Here are some useful resourses: https://www.aviasales.ru, https://www.google.com, https://www.youtube.com, https://yandex.ru/maps/org/yozh_ustritsa/52393193425/"
    result = await link_checker.run(text)
    print("result: ", result.data)
    corrected_text = await corrector.run(text, deps=result.data)
    print("corrected_text:  ", corrected_text.data)


if __name__ == "__main__":
    asyncio.run(main())
