import os


def handler(event, context):
    """Infrastructure-only placeholder. No crawling is implemented yet."""
    return {
        "statusCode": 501,
        "crawler": os.environ.get("CRAWLER_NAME", "unknown"),
        "message": "Crawler implementation intentionally deferred.",
    }

