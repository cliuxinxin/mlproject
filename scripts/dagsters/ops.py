import os
from dagster import job, op, get_dagster_logger


@op
def hello():
    get_dagster_logger().info(f"Hello, world!")

@job
def hello_job():
    hello()