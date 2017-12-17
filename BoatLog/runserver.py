"""
This script runs the BoatLog application using a development server.
"""

from os import environ
from BoatLog import app
import GraphObj

if __name__ == '__main__':
    app.secret_key = 'NickGrattan123'
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT,debug=True)
