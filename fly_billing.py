"""Fly.io GraphQL API로 현재 청구 잔액 조회"""
import logging
import httpx

logger = logging.getLogger(__name__)

_GQL_URL = "https://api.fly.io/graphql"

_BILLING_QUERY = """
query {
  organization(slug: "personal") {
    name
    billingSummary {
      currentBalance { amount currency }
      pendingBalance  { amount currency }
    }
  }
}
"""

_VERIFY_QUERY = """
query {
  viewer {
    ... on User { email name }
  }
}
"""


async def get_fly_billing(token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        # 1차: 잔액 조회
        try:
            resp = await client.post(_GQL_URL, headers=headers, json={"query": _BILLING_QUERY})
            resp.raise_for_status()
            data = resp.json()
            if "errors" not in data:
                org     = (data.get("data") or {}).get("organization") or {}
                summary = org.get("billingSummary") or {}
                current = summary.get("currentBalance") or {}
                pending = summary.get("pendingBalance") or {}
                if current.get("amount") is not None:
                    return {
                        "current_balance_usd": float(current["amount"]),
                        "pending_balance_usd": float(pending.get("amount", 0)),
                        "org": org.get("name", "Personal"),
                        "error": None,
                    }
        except Exception as e:
            logger.warning(f"Fly.io billing query failed: {e}")

        # 2차: 토큰 유효성만 확인
        try:
            resp = await client.post(_GQL_URL, headers=headers, json={"query": _VERIFY_QUERY})
            resp.raise_for_status()
            viewer = ((resp.json().get("data") or {}).get("viewer")) or {}
            return {
                "current_balance_usd": None,
                "pending_balance_usd": None,
                "org": viewer.get("name", "authenticated"),
                "error": "billing_unavailable",
            }
        except Exception as e:
            return {"error": str(e)}
