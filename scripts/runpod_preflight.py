#!/usr/bin/env python3
"""Read-only access check. No mutations, pod creation, or credential persistence."""
import json
import os
import sys
import urllib.error
import urllib.request


def query(document, key):
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": document}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def main():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print(json.dumps({"access_verified": False, "reason": "RUNPOD_API_KEY is not injected",
                          "paid_resources_created": False}))
        return 2
    try:
        result = query("query { myself { id } }", key)
        if result.get("errors") or not result.get("data", {}).get("myself", {}).get("id"):
            print(json.dumps({"access_verified": False,
                              "reason": "Account query was rejected; response body withheld",
                              "paid_resources_created": False}))
            return 1
        print(json.dumps({"access_verified": True, "paid_resources_created": False}))
        catalog = query("query { gpuTypes { id displayName memoryInGb securePrice communityPrice } }", key)
        if catalog.get("errors"):
            print(json.dumps({"catalog_verified": False, "reason": "Catalog schema or access rejected"}))
            return 1
        for gpu in catalog.get("data", {}).get("gpuTypes", []):
            if "5090" in str(gpu.get("displayName", "")):
                print(json.dumps({"gpu": gpu.get("displayName"), "id": gpu.get("id"),
                                  "memory_gib": gpu.get("memoryInGb"),
                                  "secure_price_usd_hour": gpu.get("securePrice"),
                                  "community_price_usd_hour": gpu.get("communityPrice"),
                                  "note": "Catalog prices are not an availability or final billing quote"}))
        return 0
    except urllib.error.HTTPError as error:
        print(json.dumps({"access_verified": False, "http_status": error.code,
                          "reason": "HTTP request rejected; response body withheld",
                          "paid_resources_created": False}))
        return 1
    except (urllib.error.URLError, TimeoutError):
        print(json.dumps({"access_verified": False,
                          "reason": "Network connection failed; details withheld",
                          "paid_resources_created": False}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
