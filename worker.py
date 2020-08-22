""" Worker dyno set up. Allows for heavy processes to run in background
Only works if you've subscribed to standard plan on heroku. The free subscription limits 
to only one dyno which is the web"""

# Libraries + imports
import os
from worker import conn
import redis
from rq import Worker, Queue, Connection
from src.pages.pred import write, 

# create connection
listen = ['high', 'default', 'low']
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
q = Queue(connection=conn)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()


# make a request for the pred script.
result = q.enqueue(write, 'http://heroku.com')
