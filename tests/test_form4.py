from options_copilot.edgar import FilingRef
from options_copilot.form4 import parse_form4, raw_xml_url


def doc(plan: bool, rows):
    tx = "".join(f"""<nonDerivativeTransaction><transactionCoding><transactionCode>{c}</transactionCode>
        </transactionCoding><transactionAmounts><transactionShares><value>{n}</value></transactionShares>
        </transactionAmounts></nonDerivativeTransaction>""" for c, n in rows)
    return f"""<?xml version="1.0"?><ownershipDocument><aff10b5One>{1 if plan else 0}</aff10b5One>
        <nonDerivativeTable>{tx}</nonDerivativeTable></ownershipDocument>"""


def test_counts_only_discretionary_open_market_trades():
    t = parse_form4(doc(False, [("P", 100), ("S", 250), ("F", 999), ("M", 500), ("G", 10)]))
    assert (t.bought, t.sold, t.planned_sold) == (100, 250, 0)


def test_10b5_1_sales_are_reported_separately():
    t = parse_form4(doc(True, [("S", 400)]))
    assert (t.sold, t.planned_sold) == (0, 400)


def test_raw_xml_url_strips_the_xsl_view():
    ref = FilingRef("TICKER_A", 1, "4", "0000000001-26-000001", 20261001, "xslF345X06/form4.xml")
    assert raw_xml_url(ref).endswith("/000000000126000001/form4.xml")
