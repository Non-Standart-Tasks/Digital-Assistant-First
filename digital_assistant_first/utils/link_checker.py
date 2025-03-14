from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import List
import httpx
from dotenv import load_dotenv
import socket
import requests
from urllib.parse import urlparse
import warnings
import concurrent.futures
import asyncio

# logfire.configure()

load_dotenv()


class LinkStatus(BaseModel):
    link: str = Field(description="The link to check")
    status: bool = Field(description="The status of the link")


class LinkStatusList(BaseModel):
    links: List[LinkStatus] = Field(description="The list of links to check")


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
    ),
    deps_type=LinkStatusList,
    result_type=str,
    retries=6,
)


@corrector.tool
async def get_invalid_links(ctx: RunContext[LinkStatusList]) -> List[str]:
    return [link.link for link in ctx.deps.links if not link.status]


# @link_checker.tool_plain
# async def check_link_list(list_of_links: List[str]) -> LinkStatusList:
#     """
#     Check the validity of links extracted from text
#     """
#     status = []
#     async with httpx.AsyncClient() as client:
#         for link in list_of_links:
#             try:
#                 response = await client.get(link)
#                 status.append(LinkStatus(link=link, status=response.status_code == 200))
#             except httpx.RequestError:
#                 status.append(LinkStatus(link=link, status=False))
#     return LinkStatusList(links=status)


def check_website_availability(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = "http://" + url
        parsed_url = urlparse(url)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    timeout_connect = 1.5
    timeout_read = 1.5

    # HEAD request
    try:
        response = requests.head(
            url,
            headers=headers,
            timeout=(timeout_connect, timeout_read),
            allow_redirects=True,
            verify=False,
        )

        if response.status_code < 400:
            return True, url
    except requests.exceptions.RequestException:
        pass

    # TCP check
    try:
        hostname = parsed_url.hostname
        if parsed_url.port:
            port = parsed_url.port
        else:
            port = 443 if parsed_url.scheme == "https" else 80

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)

        result = sock.connect_ex((hostname, port))
        sock.close()

        if result == 0:
            return True, url
    except Exception:
        pass

    return False, url

def check_links(links_list, max_workers=5):
    results = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(check_website_availability, url): url
                for url in links_list
            }

            for future in concurrent.futures.as_completed(future_to_url):
                status, url = future.result()
                results[url] = status

    return results


@link_checker.tool_plain
async def check_link_list(list_of_links: List[str]) -> LinkStatusList:
    """
    Check the validity of links extracted from text
    """
    results = check_links(list_of_links)

    status = []
    for url, link_status in results.items():
        status.append(LinkStatus(link=url, status=link_status))

    return LinkStatusList(links=status)

#################### Tests ####################

# links_list = [
#     "https://www.aviasales.ru/search/MOW2003LON23031",
#     "https://mycarrental.ru/",
#     "https://www.rentalcars.com/",
#     "https://www.sixt.com/",
#     "https://europcar.ru/",
#     "https://www.hertz.ru/rentacar/reservation/",
#     "https://iway.ru/",
#     "https://kiwitaxi.com/",
#     "https://wheely.com/ru",
#     "https://vip-zal.ru/",
#     "https://www.aviasales.ru/",
#     "https://rasp.yandex.ru/",
#     "https://www.aeroflot.ru/",
#     "https://www.s7.ru/",
#     "https://www.utair.ru/",
#     "https://www.uralairlines.ru/",
#     "https://ikar.aero/ru/",
#     "https://www.azurair.ru/",
#     "https://www.red-wings.ru/",
#     "https://www.turkishairlines.com/",
#     "https://www.emirates.com/my/english/book/",
#     "https://www.qatarairways.com/",
#     "https://www.etihad.com/",
#     "https://flyone.eu/",
#     "https://www.airarabia.com/en/fleet",
#     "https://www.scat.kz/",
#     "https://www.uzairways.com/",
#     "https://www.flydubai.com/",
#     "https://www.ethiopianairlines.com/my",
#     "https://southwindairlines.com/ru",
#     "https://bus.tutu.ru/",
#     "https://global.flixbus.com/",
#     "https://www.rzd.ru/",
#     "https://www.tutu.ru/",
#     "https://experience.tripster.ru/",
#     "https://www.sputnik8.com/ru/st-petersburg",
#     "https://www.viator.com/",
#     "https://www.getyourguide.com/",
#     "https://needguide.ru/",
#     "https://www.tripadvisor.ru/",
#     "https://www.rome2rio.com/",
#     "https://mirpass.vamprivet.ru/register/",
#     "https://persona.aero/services/",
#     "https://ostrovok.ru/",
#     "https://www.booking.com/",
#     "https://www.luxe.ru/",
#     "https://www.sodis.ru/",
#     "https://www.sutochno.ru/?auth=1",
#     "https://www.avito.ru/",
#     "https://tvil.ru/",
#     "https://www.alean.ru/",
#     "https://azimuthotels.com/en",
#     "https://www.mandarinoriental.com/en",
#     "https://www.fourseasons.com/",
#     "https://www.marriott.com/hotel-search.mi",
#     "https://www.jumeirah.com/en/booking/hotel-booking",
#     "https://www.ihg.com/",
#     "https://www.kerzner.com/",
#     "https://www.hilton.com/en/",
#     "https://www.shangri-la.com/",
#     "https://www.hyatt.com/",
#     "https://www.peninsula.com/en/default",
#     "https://www.baglionihotels.com/",
#     "https://www.italianhospitalitycollection.com/",
#     "https://www.oetkercollection.com/",
#     "https://cosmosgroup.ru/",
#     "https://login.accor.com/oidc/",
#     "https://accor.ru/",
#     "https://www.belmond.com/",
#     "https://www.rosewoodhotels.com/",
#     "https://www.radissonhotels.com/",
#     "https://www.virtuoso.com/",
#     "https://preferredhotels.com/",
#     "https://slh.com/",
#     "https://www.lhw.com/leaders-club",
#     "https://top-signature.com/",
#     "https://www.onlinkservices.com/",
#     "https://www.luxury-hotels.ru/",
#     "https://www.namen.ru/ru/index.html",
#     "https://www.tm-russia.ru/",
#     "https://ars-vitae.cy/",
#     "https://marketingim.ru/",
# ]

# async def main():
#     text = "Here are some useful resourses: https://www.google.com, https://www.youtube.com, https://europcar.ru/, https://www.aviasales.ru/, https://yandex.ru/maps/org/yozh_ustritsa/52393193425/"
#     result = await link_checker.run(text)
#     print(result.data)
#     corrected_text = await corrector.run(text, deps=result.data)
#     print(corrected_text.data)


# if __name__ == "__main__":
#     asyncio.run(main())
