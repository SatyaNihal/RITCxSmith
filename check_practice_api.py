"""Read the current RIT practice case and available symbols."""

import argparse
from getpass import getpass

import requests


PRACTICE_PORTS = {"etf": 16635, "volatility": 16625}


def read_json(session, base_url, resource):
    response = session.get(f"{base_url}/{resource}", timeout=10, allow_redirects=False)
    if response.status_code == 401:
        raise ValueError("RIT rejected the Trader ID or password for this case.")
    if response.status_code != 200:
        raise ValueError(f"RIT returned HTTP {response.status_code} for {resource}.")
    return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=PRACTICE_PORTS, help="Practice case to check")
    args = parser.parse_args()
    endpoint = f"http://flserver.rotman.utoronto.ca:{PRACTICE_PORTS[args.case]}/v1"

    print(f"Using RIT's practice endpoint over HTTP at {endpoint}")
    trader_id = input("Trader ID: ").strip()
    password = getpass("RIT password: ")
    if not trader_id or not password:
        parser.exit(1, "A Trader ID and password are required.\n")

    try:
        with requests.Session() as session:
            session.trust_env = False
            session.auth = (trader_id, password)
            case = read_json(session, endpoint, "case")
            securities = read_json(session, endpoint, "securities")
            tickers = ", ".join(security["ticker"] for security in securities)
            print(f"{case['name']} | status {case['status']} | tick {case['tick']}")
            print(f"Symbols: {tickers}")
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Practice API check failed: {error}\n")


if __name__ == "__main__":
    main()
